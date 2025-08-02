import argparse
import multiprocessing
from pathlib import Path
from time import time
from typing import Tuple

import ml
import ROOT
from distributed import Client, LocalCluster, SSHCluster, get_worker
from plotting import save_ml_plots, save_plots
from statistical import fit_histograms
from utils import AGCInput, AGCResult, postprocess_results, retrieve_inputs, save_histos

XSEC_INFO = {
    "ttbar": 396.87 + 332.97,
    "single_top_s_chan": 2.0268 + 1.2676,
    "single_top_t_chan": (36.993 + 22.175) / 0.252,
    "single_top_tW": 37.936 + 37.906,
    "wjets": 61457 * 0.252,
    "zprimet": 0.3086, #guess by DIMA
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AGC analysis, optionally on a single file")
    p.add_argument(
        "--n-max-files-per-sample",
        "-n",
        help="How many files per sample to process (if absent, all files).",
        type=int,
    )
    p.add_argument(
        "--data-cache",
        "-d",
        help="Directory for local data cache.",
    )
    p.add_argument(
        "--remote-data-prefix",
        help="Replace default remote prefix[](https://xrootd-local.unl.edu:1094//store/user/AGC) when accessing data.",
    )
    p.add_argument(
        "--output",
        "-o",
        help="Output ROOT file name.",
        default="histograms.root",
    )
    p.add_argument(
        "--inference",
        action=argparse.BooleanOptionalAction,
        help="Enable machine learning histograms.",
    )
    p.add_argument(
        "--scheduler",
        "-s",
        help="RDataFrame scheduler.",
        default="mt",
        choices=["mt", "dask-local", "dask-ssh", "dask-remote"],
    )
    p.add_argument(
        "--scheduler-address",
        help="Dask scheduler address (required for dask-remote).",
    )
    p.add_argument(
        "--ncores",
        "-c",
        help="Number of cores to use.",
        default=multiprocessing.cpu_count(),
        type=int,
    )
    p.add_argument(
        "--npartitions",
        help="Number of data partitions for distributed execution.",
        type=int,
    )
    p.add_argument(
        "--hosts",
        help="Comma-separated list of worker node hostnames for dask-ssh.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose logs.",
        action="store_true",
    )
    p.add_argument(
        "--statistical-validation",
        help=argparse.SUPPRESS,
        action="store_true",
    )
    p.add_argument(
        "--no-fitting",
        help="Skip statistical validation.",
        action="store_true",
    )
    p.add_argument(
        "--sample",
        help="Sample to process (e.g., ttbar). Required if --file-name is specified.",
    )
    p.add_argument(
        "--file-name",
        help="Specific file to process (e.g., file1.root or full URL).",
    )
    return p.parse_args()

def create_dask_client(scheduler: str, ncores: int, hosts: str, scheduler_address: str) -> Client:
    if scheduler == "dask-local":
        lc = LocalCluster(n_workers=ncores, threads_per_worker=1, processes=True)
        return Client(lc)
    if scheduler == "dask-ssh":
        workers = hosts.split(",")
        print(f"Using worker nodes: {workers=}")
        sshc = SSHCluster(
            workers,
            connect_options={"known_hosts": None},
            worker_options={
                "nprocs": ncores,
                "nthreads": 1,
                "memory_limit": "32GB",
            },
        )
        return Client(sshc)
    if scheduler == "dask-remote":
        return Client(scheduler_address)
    raise ValueError(
        f"Unexpected scheduling mode '{scheduler}'. Valid modes are ['dask-local', 'dask-ssh', 'dask-remote']."
    )

def define_trijet_mass(df: ROOT.RDataFrame) -> ROOT.RDataFrame:
    df = df.Filter("Sum(Jet_btagCSVV2_cut > 0.5) > 1")
    df = df.Define(
        "Jet_p4",
        "ConstructP4(Jet_pt_cut, Jet_eta_cut, Jet_phi_cut, Jet_mass_cut)",
    )
    df = df.Define("Trijet_idx", "Combinations(Jet_pt_cut, 3)")
    df = df.Define(
        "Trijet_btag",
        """
            auto J1_btagCSVV2 = Take(Jet_btagCSVV2_cut, Trijet_idx[0]);
            auto J2_btagCSVV2 = Take(Jet_btagCSVV2_cut, Trijet_idx[1]);
            auto J3_btagCSVV2 = Take(Jet_btagCSVV2_cut, Trijet_idx[2]);
            return J1_btagCSVV2 > 0.5 || J2_btagCSVV2 > 0.5 || J3_btagCSVV2 > 0.5;
            """,
    )
    df = df.Define(
        "Trijet_p4",
        """
        auto J1 = Take(Jet_p4, Trijet_idx[0]);
        auto J2 = Take(Jet_p4, Trijet_idx[1]);
        auto J3 = Take(Jet_p4, Trijet_idx[2]);
        return (J1+J2+J3)[Trijet_btag];
        """,
    )
    df = df.Define(
        "Trijet_pt",
        "return Map(Trijet_p4, [](const ROOT::Math::PxPyPzMVector &v) { return v.Pt(); })",
    )
    df = df.Define("Trijet_mass", "Trijet_p4[ArgMax(Trijet_pt)].M()")
    return df

def book_histos(
    df: ROOT.RDataFrame, process: str, variation: str, nevents: int, inference=False
) -> Tuple[list[AGCResult], list[AGCResult]]:
    x_sec = XSEC_INFO[process]
    lumi = 3378  # /pb
    xsec_weight = x_sec * lumi / nevents
    df = df.Define("Weights", str(xsec_weight))

    if variation == "nominal":
        df = df.Vary(
            "Jet_pt",
            "ROOT::RVec<ROOT::RVecF>{Jet_pt*pt_scale_up(), Jet_pt*jet_pt_resolution(Jet_pt)}",
            ["pt_scale_up", "pt_res_up"],
        )
        if process == "wjets":
            df = df.Vary(
                "Weights",
                "Weights*flat_variation()",
                [f"scale_var_{direction}" for direction in ["up", "down"]],
            )

    df = (
        df.Define(
            "Electron_mask",
            "Electron_pt > 30 && abs(Electron_eta) < 2.1 && Electron_sip3d < 4 && Electron_cutBased == 4",
        )
        .Define(
            "Muon_mask",
            "Muon_pt > 30 && abs(Muon_eta) < 2.1 && Muon_sip3d < 4 && Muon_tightId && Muon_pfRelIso04_all < 0.15",
        )
        .Filter("Sum(Electron_mask) + Sum(Muon_mask) == 1")
        .Define("Jet_mask", "Jet_pt > 30 && abs(Jet_eta) < 2.4 && Jet_jetId == 6")
        .Filter("Sum(Jet_mask) >= 4")
    )
    df = (
        df.Define("Jet_pt_cut", "Jet_pt[Jet_mask]")
        .Define("Jet_btagCSVV2_cut", "Jet_btagCSVV2[Jet_mask]")
        .Define("Jet_eta_cut", "Jet_eta[Jet_mask]")
        .Define("Jet_phi_cut", "Jet_phi[Jet_mask]")
        .Define("Jet_mass_cut", "Jet_mass[Jet_mask]")
    )
    if variation == "nominal":
        df = df.Vary(
            "Weights",
            "ROOT::RVecD{Weights*btag_weight_variation(Jet_pt_cut)}",
            [
                f"{weight_name}_{direction}"
                for weight_name in [f"btag_var_{i}" for i in range(4)]
                for direction in ["up", "down"]
            ],
        )

    df4j1b = df.Filter("Sum(Jet_btagCSVV2_cut > 0.5) == 1").Define("HT", "Sum(Jet_pt_cut)")
    df4j2b = define_trijet_mass(df)

    results = []
    for df, observable, region in zip([df4j1b, df4j2b], ["HT", "Trijet_mass"], ["4j1b", "4j2b"]):
        histo_model = ROOT.RDF.TH1DModel(
            name=f"{region}_{process}_{variation}",
            title=process,
            nbinsx=25,
            xlow=50,
            xup=550,
        )
        nominal_histo = df.Histo1D(histo_model, observable, "Weights")
        results.append(
            AGCResult(
                nominal_histo,
                region,
                process,
                variation,
                nominal_histo,
                should_vary=(variation == "nominal"),
            )
        )
        print(f"Booked histogram {histo_model.fName}")

    ml_results: list[AGCResult] = []
    if inference:
        df4j2b = ml.define_features(df4j2b)
        df4j2b = ml.infer_output_ml_features(df4j2b)
        for i, feature in enumerate(ml.ml_features_config):
            histo_model = ROOT.RDF.TH1DModel(
                name=f"{feature.name}_{process}_{variation}",
                title=feature.title,
                nbinsx=feature.binning[0],
                xlow=feature.binning[1],
                xup=feature.binning[2],
            )
            nominal_histo = df4j2b.Histo1D(histo_model, f"results{i}", "Weights")
            ml_results.append(
                AGCResult(
                    nominal_histo,
                    feature.name,
                    process,
                    variation,
                    nominal_histo,
                    should_vary=(variation == "nominal"),
                )
            )
            print(f"Booked histogram {histo_model.fName}")

    return (results, ml_results)

def load_cpp():
    try:
        localdir = get_worker().local_directory
        cpp_source = Path(localdir) / "helpers.h"
    except ValueError:
        cpp_source = "helpers.h"
    ROOT.gInterpreter.Declare(f'#include "{str(cpp_source)}"')

def run_mt(
    program_start: float,
    args: argparse.Namespace,
    inputs: list[AGCInput],
    results: list[AGCResult],
    ml_results: list[AGCResult],
) -> None:
    ROOT.EnableImplicitMT(args.ncores)
    print(f"Number of threads: {ROOT.GetThreadPoolSize()}")
    load_cpp()
    if args.inference:
        ml.load_cpp()

    for input in inputs:
        df = ROOT.RDataFrame("Events", input.paths)
        hist_list, ml_hist_list = book_histos(
            df, input.process, input.variation, input.nevents, inference=args.inference
        )
        results += hist_list
        ml_results += ml_hist_list

    for r in results + ml_results:
        if r.should_vary:
            r.histo = ROOT.RDF.Experimental.VariationsFor(r.histo)

    print(f"Building the computation graphs took {time() - program_start:.2f} seconds")
    run_graphs_start = time()
    ROOT.RDF.RunGraphs([r.nominal_histo for r in results + ml_results])
    print(f"Executing the computation graphs took {time() - run_graphs_start:.2f} seconds")

def run_distributed(
    program_start: float,
    args: argparse.Namespace,
    inputs: list[AGCInput],
    results: list[AGCResult],
    ml_results: list[AGCResult],
) -> None:
    if args.inference:
        def ml_init():
            load_cpp()
            ml.load_cpp()
        ROOT.RDF.Experimental.Distributed.initialize(ml_init)
    else:
        ROOT.RDF.Experimental.Distributed.initialize(load_cpp)

    scheduler_address = args.scheduler_address if args.scheduler_address else ""
    with create_dask_client(args.scheduler, args.ncores, args.hosts, scheduler_address) as client:
        for input in inputs:
            df = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame(
                "Events",
                input.paths,
                daskclient=client,
                npartitions=args.npartitions,
            )
            df._headnode.backend.distribute_unique_paths(
                [
                    "helpers.h",
                    "ml_helpers.h",
                    "ml.py",
                    "models/bdt_even.root",
                    "models/bdt_odd.root",
                ]
            )
            hist_list, ml_hist_list = book_histos(
                df, input.process, input.variation, input.nevents, inference=args.inference
            )
            results += hist_list
            ml_results += ml_hist_list

        for r in results + ml_results:
            if r.should_vary:
                r.histo = ROOT.RDF.Experimental.Distributed.VariationsFor(r.histo)

        print(f"Building the computation graphs took {time() - program_start:.2f} seconds")
        run_graphs_start = time()
        ROOT.RDF.Experimental.Distributed.RunGraphs([r.nominal_histo for r in results + ml_results])
        print(f"Executing the computation graphs took {time() - run_graphs_start:.2f} seconds")

def main() -> None:
    program_start = time()
    args = parse_args()

    ROOT.TH1.AddDirectory(False)
    ROOT.gROOT.SetBatch(True)

    if args.verbose:
        ROOT.Detail.RDF.RDFLogChannel().SetVerbosity(ROOT.Experimental.ELogLevel.kInfo)

    if args.statistical_validation:
        fit_histograms(filename=args.output)
        return

    if args.file_name and args.n_max_files_per_sample:
        logging.warning("--file-name overrides --n-max-files-per-sample")
        args.n_max_files_per_sample = None

    inputs: list[AGCInput] = retrieve_inputs(
        args.n_max_files_per_sample,
        args.remote_data_prefix,
        args.data_cache,
        sample=args.sample,
        file_name=args.file_name,
    )
    if not inputs:
        raise RuntimeError("No input files selected for analysis")

    for input in inputs:
        print(f"Processing {input.process} ({input.variation}) with {input.nevents} events from {input.paths}")

    results: list[AGCResult] = []
    ml_results: list[AGCResult] = []

    if args.scheduler == "mt":
        run_mt(program_start, args, inputs, results, ml_results)
    else:
        if args.scheduler == "dask-remote" and not args.scheduler_address:
            raise ValueError("'dask-remote' chosen but no scheduler address provided")
        if args.scheduler_address and args.scheduler != "dask-remote":
            raise ValueError("Scheduler address provided but scheduler is not 'dask-remote'")
        run_distributed(program_start, args, inputs, results, ml_results)

    results = postprocess_results(results)
    
    save_plots(results)
    save_histos([r.histo for r in results], output_fname=args.output)
    print(f"Result histograms saved in file {args.output}")

    if args.inference:
        ml_results = postprocess_results(ml_results)
        save_ml_plots(ml_results)
        output_fname = args.output.split(".root")[0] + "_ml_inference.root"
        save_histos([r.histo for r in ml_results], output_fname=output_fname)
        print(f"Result histograms from ML inference step saved in file {output_fname}")

    if not args.no_fitting:
        fit_histograms(filename=args.output)

if __name__ == "__main__":
    main()