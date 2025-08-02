import ROOT
from ml import ml_features_config
from utils import AGCResult


def save_plots(results: list[AGCResult], output_dir: str = "results"):
    import ROOT
    import os
    width = 2160
    height = 2160
    c = ROOT.TCanvas("c", "c", width, height)
    ROOT.gStyle.SetPalette(ROOT.kRainBow)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Region 1 stack (4j1b, nominal)
    hlist = [r.histo for r in results if r.region == "4j1b" and r.variation == "nominal" and r.histo and r.histo.GetEntries() > 0]
    if hlist:
        hlist = [h.Clone().Rebin(2) for h in hlist]
        hs = ROOT.THStack("j4b1", ">=4 jets, 1 b-tag; H_{T} [GeV]")
        for h in hlist:
            hs.Add(h)
        hs.Draw("hist pfc plc")
        x_axis = hs.GetXaxis()
        if x_axis:
            x_axis.SetRangeUser(120, x_axis.GetXmax())
            x_axis.SetTitleOffset(1.5)
            x_axis.CenterTitle()
        c.BuildLegend(0.65, 0.7, 0.9, 0.9)
        c.Draw()
        c.SaveAs(os.path.join(output_dir, "reg1.png"))
    else:
        print("Warning: No valid histograms for region '4j1b', variation 'nominal'")

    # Region 2 stack (4j2b, nominal)
    hlist = [r.histo for r in results if r.region == "4j2b" and r.variation == "nominal" and r.histo and r.histo.GetEntries() > 0]
    if hlist:
        hs = ROOT.THStack("j4b2", ">=4 jets, 2 b-tag; m_{bjj} [GeV]")
        for h in hlist:
            hs.Add(h)
        hs.Draw("hist pfc plc")
        x_axis = hs.GetXaxis()
        if x_axis:
            x_axis.SetTitleOffset(1.5)
            x_axis.CenterTitle()
        c.BuildLegend(0.65, 0.7, 0.9, 0.9)
        c.Draw()
        c.SaveAs(os.path.join(output_dir, "reg2.png"))
    else:
        print("Warning: No valid histograms for region '4j2b', variation 'nominal'")

    # b-tag variations (4j1b, ttbar only)
    btag_variations = ["nominal", "btag_var_0_up", "btag_var_1_up", "btag_var_2_up", "btag_var_3_up"]
    hlist = [r.histo for r in results if r.region == "4j1b" and r.process == "ttbar" and r.variation in btag_variations and r.histo and r.histo.GetEntries() > 0]
    if hlist:
        hlist = [h.Clone().Rebin(2) for h in hlist]
        hs = ROOT.THStack("j4b1btag", "btag-variations ; H_{T} [GeV]")
        for h, name in zip(hlist, btag_variations):
            h.SetLineWidth(4)
            h.SetTitle(name)
            hs.Add(h)
        hs.Draw("hist nostack plc")
        x_axis = hs.GetXaxis()
        if x_axis:
            x_axis.SetRangeUser(120, x_axis.GetXmax())
            x_axis.SetTitleOffset(1.5)
            x_axis.CenterTitle()
        c.BuildLegend(0.65, 0.7, 0.9, 0.9)
        c.Draw()
        c.SaveAs(os.path.join(output_dir, "btag.png"))
    else:
        print("Warning: No valid histograms for b-tag variations (4j1b, ttbar)")

    # Jet energy variations (4j2b, ttbar only)
    jet_variations = ["nominal", "pt_scale_up", "pt_res_up"]
    hlist = [r.histo for r in results if r.region == "4j2b" and r.process == "ttbar" and r.variation in jet_variations and r.histo and r.histo.GetEntries() > 0]
    if hlist:
        hs = ROOT.THStack("4j2bjet", "Jet energy variations ; m_{bjj} [GeV]")
        for h, name in zip(hlist, jet_variations):
            h.SetFillColor(0)
            h.SetLineWidth(4)
            h.SetTitle(name)
            hs.Add(h)
        hs.Draw("hist nostack plc")
        x_axis = hs.GetXaxis()
        if x_axis:
            x_axis.SetRangeUser(0, 600)
            x_axis.SetTitleOffset(1.5)
            x_axis.CenterTitle()
        c.BuildLegend(0.65, 0.7, 0.9, 0.9)
        c.Draw()
        c.SaveAs(os.path.join(output_dir, "jet.png"))
    else:
        print("Warning: No valid histograms for jet energy variations (4j2b, ttbar)")


def save_ml_plots(results: list[AGCResult]):
    width = 2160
    height = 2160
    c = ROOT.TCanvas("c", "c", width, height)

    for i, feature in enumerate(ml_features_config):
        hlist = [r.histo for r in results if r.variation == "nominal" and r.region == feature]
        hs = ROOT.THStack("features", feature.title)
        for h in hlist:
            hs.Add(h)
        hs.Draw("hist pfc plc")
        c.BuildLegend()
        c.Print("features.pdf" + (i == 0) * "(" + (i + 1 == len(ml_features_config)) * ")")
