import pandas as pd, os, matplotlib
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image as RLImage, KeepTogether, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FD = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FD, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FD, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FD, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(FD, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

OUT = "/mnt/user-data/outputs/output"
s = pd.read_csv(f"{OUT}/OEX_calibration_summary.csv")
c = pd.read_csv(f"{OUT}/OEX_term_structure_comparison.csv")
mc = pd.read_csv(f"{OUT}/OEX_expansion_vs_MC_check.csv")

body = ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.6, spaceAfter=6, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#333333"), spaceAfter=4)
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=22, leading=26, spaceAfter=4)
meta = ParagraphStyle("meta", parent=body, fontSize=10, textColor=colors.HexColor("#444444"), spaceAfter=2)
bl = ParagraphStyle("bl", parent=body, spaceAfter=3)

P = lambda t, st=body: Paragraph(t, st)
def bullets(items, st=bl):
    return ListFlowable([ListItem(P(i, st), leftIndent=12) for i in items], bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV", bulletFontSize=8)
def numbered(items):
    return ListFlowable([ListItem(P(i, bl), leftIndent=16) for i in items], bulletType="1", leftIndent=16, bulletFontName="DV", bulletFontSize=9)

def table(data, widths, header=True, align_right_from=None, fs=7.4):
    rows = []
    for ri, row in enumerate(data):
        rows.append([P(str(x), cellb if (header and ri == 0) else cell) for x in row])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF3")))
    t.setStyle(TableStyle(st)); return t

pct = lambda x: f"{100*x:.2f}"
names = list(s.Period)
W = 7.0 * inch
story = []

# ---------------------------------------------------------------- title
story += [P("Final Output", title), P("Yuto Sugiyama", meta), P("Tuesday, October 27, 2026", meta), Spacer(1, 6),
          P("Summary: model-implied versus market volatility surface (OEX), maturities up to 1 year", h1),
          P("This version extends the earlier summary by adding a <b>1-year (252 trading-day) maturity</b> to the whole analysis: "
            "the tuning targets, the model surface, the tables and the figures. Everything else (periods, model, fixed parameters) is unchanged.", body)]

# ---------------------------------------------------------------- what the code does
story += [P("What the code does", h2),
  P("The script <b>OEX_surface_comparison.py</b> builds a model-implied volatility surface for the OEX in each period and compares it with the market "
    "volatility data. It reuses the model functions in SurfaceVolatility.py without changing them (the perturbation expansion for a stochastic-volatility "
    "model after Diehl, then Black-Scholes inversion). The steps are:", body),
  numbered([
   "<b>Load the data.</b> The two OEX historical-volatility files are combined (price and 10, 20, 30, 60, 90, 120, 150 and 180 trading-day historical volatility). "
   "The files contain no 1-year column, so the <b>252-day historical volatility is computed from the price series</b>: the sample standard deviation of daily log returns times √252. "
   "This formula reproduces the supplied 10-, 60- and 180-day columns exactly (to machine precision), so the new tenor is consistent with the others. "
   "The USD interest-rate file is converted from percent to decimals.",
   "<b>Cut the sample into periods</b> (listed below).",
   "<b>Build the market surface.</b> For each period, the historical volatility is averaged at each of the nine tenors (10 to 252 trading days), "
   "<b>over the same dates for every tenor</b> (rows where all nine exist). Because the data contain no option prices, this surface is flat in strike, so the market side is a term structure and not a true implied-volatility surface.",
   "<b>Tune the model.</b> Three model parameters are tuned so that the model's at-the-money implied volatility follows the nine-point market term structure (details below).",
   "<b>Build the model surface.</b> For 17 maturities (10 to 252 trading days, 0.04 to 1.00 years) and 41 strikes (90% to 110% of spot), the script computes the expansion price, "
   "then inverts Black-Scholes to get the implied volatility. The interest rate is the period-average USD curve at each maturity, spot is the last price in the period, and there is no dividend yield. "
   "Points where |ln(K/S<sub>0</sub>)| &gt; 2.5 σ√T are left out because the expansion is unreliable there.",
   "<b>Compare and save</b> the figures and tables described under Outputs."])]

# ---------------------------------------------------------------- periods
story += [P("Periods used", h2),
  P("The full sample runs from 2016-01-04 to 2026-09-30. The pre-COVID window (2016 to 2019) is kept as a single period, and the post-COVID years are split into four regimes based on the level and shape of historical volatility. "
    "<b>One change from the earlier version:</b> 252-day historical volatility cannot be computed before 2017-01-03 (it needs 252 prior returns), so the pre-COVID averages now cover 2017-01-03 to 2019-12-31 "
    "for all tenors. Using the same dates for every tenor matters: otherwise the shorter tenors would include the higher-volatility year 2016 and the curve would bend down at one year for that reason alone.", body),
  table([["Period", "Dates", "Why"],
         ["Pre-COVID", "2016-01-01 to 2019-12-31 (averages from 2017-01-03)", "Baseline: moderate volatility"],
         ["COVID shock", "2020-01-01 to 2020-12-31", "Volatility spike"],
         ["Recovery", "2021-01-01 to 2021-12-31", "Return to calm"],
         ["Tightening bear", "2022-01-01 to 2022-12-31", "High volatility, inverted term structure"],
         ["Normalisation", "2023-01-01 to 2026-09-30", "Moderate volatility again"]],
        [1.3*inch, 3.1*inch, 2.6*inch])]

# ---------------------------------------------------------------- parameters
story += [P("Model parameters: tuned and fixed", h2),
  P("<b>Tuned (per period).</b> The initial volatility V<sub>0</sub>, the long-run volatility θ and the mean-reversion speed κ are chosen by least squares so that the model's at-the-money implied volatility "
    "matches the average historical volatility at the nine tenors. They are bounded: V<sub>0</sub> ∈ [0.02, 0.8], κ ∈ [0.1, 10], and θ between 1.0 and 1.6 times the <b>longest-tenor (now 252-day)</b> historical volatility. "
    "In four of the five periods (all except 2022) θ ends at the upper bound, so it is effectively set by the bound rather than estimated from the data; V<sub>0</sub> and κ are the parameters the fit actually determines.", body),
  P("<b>Fixed (guessed, not fitted).</b> The vol-of-vol λ = 0.8 and the spot-volatility correlation ρ = −0.7 are the same in every period. A term structure of historical volatility cannot identify them. "
    "The finite-difference step in the expansion is h = 0.01 (the original script used 0.5), and the rates, spot and zero dividend yield are as described above.", body)]
t1 = [["Period", "V<sub>0</sub> (tuned)", "theta (tuned)", "kappa (tuned)", "lambda (fixed)", "rho (fixed)"]]
for _, r in s.iterrows(): t1.append([r.Period, f"{r.V0:.3f}", f"{r.theta:.3f}", f"{r.kappa:.3f}", "0.8", "-0.7"])
story += [P("Table 1: Model parameters by period (nine-tenor fit including 1 year).", cap), table(t1, [2.1*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 0.9*inch])]

# ---------------------------------------------------------------- tuning history
story += [P("How the tuning developed", h2),
  P("The tuning went through several rounds. Each round changed one thing (what is free, the bounds, the guessed vol-of-vol, the period split, or the set of maturities) after the previous round gave a fit that was either poor or implausible. "
    "Rounds 1 to 5 are unchanged from the earlier summary (eight tenors up to 180 days); round 6 is the current analysis with the 1-year tenor added. "
    "The first rows are fixed numbers from earlier runs; the round 6 rows are read from OEX_calibration_summary.csv.", body)]
hist = [["Round", "Setting", "Period", "V<sub>0</sub>", "theta", "kappa", "lambda", "ATM RMSE (vol pts)"],
 ["1", "V0, theta and kappa all free", "Pre-COVID (2016-2019)", "0.112", "0.219", "1.156", "0.8", "0.090"],
 ["1", "V0, theta and kappa all free", "Post-COVID (2020-2026)", "0.171", "0.800", "0.251", "1.0", "0.070"],
 ["2", "theta fixed at the 180-day market vol; V0, kappa free (kappa >= 0.5)", "Pre-COVID (2016-2019)", "0.126", "0.132", "0.500", "0.8", "0.970"],
 ["2", "theta fixed at the 180-day market vol; V0, kappa free (kappa >= 0.5)", "Post-COVID (2020-2026)", "0.193", "0.192", "0.500", "1.0", "1.750"],
 ["3", "theta between 1.0 and 1.6 x the 180-day market vol", "Pre-COVID (2016-2019)", "0.112", "0.211", "1.332", "0.8", "0.090"],
 ["3", "theta between 1.0 and 1.6 x the 180-day market vol", "Post-COVID (2020-2026)", "0.175", "0.307", "1.226", "1.0", "0.890"],
 ["4", "As round 3, lambda lowered to 0.8", "Post-COVID (2020-2026)", "0.170", "0.307", "1.379", "0.8", "0.330"],
 ["5", "As round 4, post-COVID split into four regimes (8 tenors, to 180 days)", "Pre-COVID (2016-2019)", "0.111", "0.211", "1.332", "0.8", "0.095"],
 ["5", "(same)", "COVID shock (2020)", "0.272", "0.507", "1.385", "0.8", "2.120"],
 ["5", "(same)", "Recovery (2021)", "0.121", "0.243", "1.251", "0.8", "0.404"],
 ["5", "(same)", "Tightening bear (2022)", "0.247", "0.280", "0.964", "0.8", "0.170"],
 ["5", "(same)", "Normalisation (2023-2026)", "0.138", "0.262", "1.413", "0.8", "0.210"]]
for i, (_, r) in enumerate(s.iterrows()):
    hist.append(["6 (final)" if i == 0 else "6 (final)", "As round 5, with the 1-year tenor added (9 tenors, common sample, theta bounds on 252-day vol)" if i == 0 else "(same)",
                 r.Period, f"{r.V0:.3f}", f"{r.theta:.3f}", f"{r.kappa:.3f}", "0.8", f"{100*r.RMSE_ATM_vs_HV:.3f}"])
story += [P("Table 2: Fitted parameters in each tuning round. Round 4 changes only the post-COVID period, so the pre-COVID row is the same as in round 3.", cap),
          table(hist, [0.55*inch, 2.0*inch, 1.5*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.65*inch, 0.8*inch])]
story += [Spacer(1, 6), bullets([
 "<b>Round 1: everything free.</b> The fit was almost exact, but the post-COVID result was not plausible: θ went to the upper bound of 0.8 (about four times the observed volatility) with a very slow κ of 0.25. With only a term structure of historical volatility to fit, a high θ combined with a slow κ can mimic the same curve as a modest θ combined with a fast κ, so the data do not separate the two.",
 "<b>Round 2: θ fixed at the long-end market volatility.</b> This removed the degeneracy but made the fit clearly worse (errors of 1 to 1.75 volatility points), and κ ran into its lower bound.",
 "<b>Round 3: θ between 1.0 and 1.6 times the long-end volatility.</b> The pre-COVID fit returned to 0.09 points and κ became sensible, but the post-COVID error was still 0.89 points because λ = 1.0 pulls the model's at-the-money volatility down at long maturities.",
 "<b>Round 4: λ = 0.8 in both periods.</b> The post-COVID error fell to 0.33 volatility points.",
 "<b>Round 5: the five regimes.</b> The fit was within 0.4 volatility points in every regime except 2020, where the cap on θ binds (error 2.1 points).",
 "<b>Round 6 (final): the 1-year tenor added.</b> The same settings, now with nine tenors. The fit is 0.20 points in pre-COVID and 2022, 0.72 in 2023-2026, 1.52 in 2021 and 3.13 in 2020. "
 "The errors that grew are almost entirely at the new 1-year point, where the market curve bends away from the smooth shape the model can produce (see below). The tuned V<sub>0</sub>, κ and θ moved more than between rounds 3 to 5 (for example κ in 2022 fell from 0.96 to 0.49), so the parameter values should be read as descriptive settings that reproduce the curve, not as estimates of the true dynamics."]),
 ]

# ---------------------------------------------------------------- outputs
story += [P("Outputs", h2),
  P("All files are in the folder <b>output</b>. They replace the earlier eight-tenor results of the same names, so keep the two versions in separate project folders.", body),
  bullets([
   "<b>OEX_surfaces_model_vs_market.png</b>: one 3D panel per period, with the model-implied surface (coloured) and the market historical-volatility surface (red, flat in strike) on common axes, now up to 1 year.",
   "<b>OEX_term_structure_and_smiles.png</b>: the market term structure against the model at-the-money implied volatility for all periods (10 to 252 days), the model smile at about three months, and a new model smile at one year.",
   "<b>OEX_calibration_summary.csv</b> and <b>OEX_term_structure_comparison.csv</b>: fitted parameters, fit error, 3-month and 1-year skew, and the market and model values at every tenor.",
   "<b>OEX_surface_&lt;period&gt;.csv</b> (five files): the full model surface by maturity and strike, with the market value alongside.",
   "<b>OEX_expansion_vs_MC_check.csv</b> (new): a Monte Carlo accuracy check of the expansion at three months and one year (see the last section)."])]

# ---------------------------------------------------------------- comparison
story += [P("Model-implied surface compared with the market surface", h1), P("Which values are model and which are market", h2)]
story.append(table([
 ["Value", "Where it is", "Model or market"],
 ["Market_HV", "OEX_term_structure_comparison.csv and the OEX_surface_&lt;period&gt;.csv files", "Market: average historical volatility over the period at that tenor (10 to 252 days). It does not change with strike."],
 ["Model_ATM_IV", "OEX_term_structure_comparison.csv", "Model: implied volatility at the money for the same tenor."],
 ["Diff", "OEX_term_structure_comparison.csv", "Comparison: model minus market."],
 ["Model_IV", "OEX_surface_&lt;period&gt;.csv", "Model: implied volatility at each maturity and strike."],
 ["RMSE_ATM_vs_HV", "OEX_calibration_summary.csv", "Comparison: the typical gap between model and market at the money across the nine tenors."],
 ["Skew_3M_IV90_minus_IV110, Skew_1Y_IV90_minus_IV110", "OEX_calibration_summary.csv", "Model only: model implied volatility at 90% moneyness minus that at 110%, at about three months and at one year. The market has no equivalent because it is flat in strike."],
 ["Market_HV_10d, Market_HV_180d, Market_HV_1y; Model_ATM_IV_1y", "OEX_calibration_summary.csv", "Market: the 10-day, 180-day and 1-year points of the market term structure. Model: the at-the-money value at one year."]],
 [1.7*inch, 2.2*inch, 3.1*inch]))
story += [Spacer(1, 4), P("In the figures, the coloured surface (and the dashed lines in the term-structure plot) is the model, and the red flat sheet (and the solid lines) is the market.", body),
  P("Which values to compare", h2),
  numbered([
   "<b>Term structure (the main comparison).</b> In OEX_term_structure_comparison.csv, compare Model_ATM_IV with Market_HV row by row, one row per period and tenor. A Diff near zero means the model matches the market. "
   "Out to 180 days the values agree closely everywhere except 2020, where Diff is −0.046 at 180 days. At 252 days the gaps are larger in 2020 (−0.048), 2021 (−0.043) and 2023-2026 (−0.020).",
   "<b>Full surface.</b> In each OEX_surface_&lt;period&gt;.csv, compare Model_IV with Market_HV within each maturity. Because Market_HV is the same at every strike, the only extra information from the model is how Model_IV varies across strikes (the smile), which is very small here.",
   "<b>One number per period.</b> RMSE_ATM_vs_HV in OEX_calibration_summary.csv summarises the gap from item 1."])]

# Table 3a / 3b
k = lambda name, ten, col: float(c[(c.Period == name) & (c.Tenor_days == ten)][col].iloc[0])
t3a = [["Period", "Market 10d (%)", "Model 10d (%)", "Market 180d (%)", "Model 180d (%)", "Market 1y (%)", "Model 1y (%)"]]
t3b = [["Period", "Model vs market: ATM RMSE, 9 tenors (vol pts)", "Largest gap up to 180d (vol pts)", "Gap at 1y, model minus market (vol pts)", "Model only: 3M skew (vol pts)", "Model only: 1Y skew (vol pts)"]]
for _, r in s.iterrows():
    n = r.Period
    t3a.append([n] + [f"{100*k(n,t,col):.2f}" for t in (10, 180, 252) for col in ("Market_HV", "Model_ATM_IV")])
    mx = c[(c.Period == n) & (c.Tenor_days <= 180)].Diff.abs().max()
    t3b.append([n, f"{100*r.RMSE_ATM_vs_HV:.2f}", f"{100*mx:.2f}", f"{100*k(n,252,'Diff'):+.2f}", f"{100*r.Skew_3M_IV90_minus_IV110:.2f}", f"{100*r.Skew_1Y_IV90_minus_IV110:.2f}"])
story += [P("The tables below put the model and market values for the shortest, 180-day and 1-year tenors side by side, together with the fit error and the model skew. The key under the tables states what each column means and whether it is a market value, a model value, or a comparison of the two.", body),
  P("Table 3a: Market and model-implied volatility by period (10-day, 180-day and 1-year tenors).", cap),
  table(t3a, [1.9*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch]), Spacer(1, 6),
  P("Table 3b: Fit error and model skew by period.", cap),
  table(t3b, [1.7*inch, 1.25*inch, 1.05*inch, 1.15*inch, 0.95*inch, 0.9*inch]), Spacer(1, 6),
  table([["Column", "What it is", "Type"],
   ["Market 10d / 180d / 1y (%)", "Average historical volatility over the period at the 10-day, 180-day and 252-day (1-year) tenors. Flat in strike.", "Market"],
   ["Model 10d / 180d / 1y (%)", "Model-implied volatility at the money (strike equal to spot) at the same tenors.", "Model"],
   ["ATM RMSE (vol pts)", "Root-mean-square difference between the model at-the-money volatility and the market volatility, taken over all nine tenors (10 to 252 days). Smaller means a closer match.", "Comparison of model and market"],
   ["Largest gap up to 180d; gap at 1y", "Largest absolute model-minus-market difference over the eight tenors up to 180 days; signed difference at the 1-year tenor.", "Comparison of model and market"],
   ["3M / 1Y skew (vol pts)", "Model-implied volatility at 90% moneyness minus that at 110%, at about three months and at one year. The market surface is flat in strike by construction, so it has no counterpart.", "Model only"]],
   [1.7*inch, 4.0*inch, 1.3*inch]),
  Spacer(1, 6),
  P("The two summary columns answer different questions. The ATM RMSE says how well the model matches the market along maturity. The skew describes the model's own shape along strike, and it cannot be compared with the market without option quotes.", body),
  bullets([
   "<b>Level.</b> Market volatility is about 11-13% before COVID, 26-32% in 2020, 13-19% in 2021, 19-25% in 2022 and 14-17% from 2023. The tuned model follows these levels because V<sub>0</sub> and θ are fitted to them.",
   "<b>Shape.</b> The market term structure slopes upward before COVID and from 2023, is humped in 2020 (31.7% at 180 days, 29.0% at one year), and is inverted in 2022 (24.3% at 10 days, 19.5% at one year). "
   "In 2021 it steepens sharply at the last tenor (15.2% at 180 days, 19.5% at one year) because the 252-day window still reaches back into the high-volatility second half of 2020; historical volatility looks backward, so this is not a forecast. "
   "The model matches the pre-COVID and 2022 shapes at all nine tenors, including the inverted curve in 2022.",
   "<b>Where it fits worst.</b> In 2020 the model overshoots the market by 3.6 points at 10 days and undershoots it by 4.6 points at 180 days and 4.8 points at one year: its at-the-money curve slopes down while the market climbs, "
   "because the steep 2020 term structure would need a θ above the cap. At the new 1-year tenor the model also falls 4.3 points short in 2021 and 2.0 points short in 2023-2026, where the market curve rises faster than the model's, which flattens toward its fitted long-run level.",
   "<b>Strike dimension.</b> The market surface has no strike dependence, but the model smile is also nearly flat: at three months, volatility at 90% moneyness exceeds that at 110% by 0.3 volatility points or less in every period, and at one year by 0.4 to 1.0 points. "
   "This follows from the expansion with the guessed λ and ρ, and it is not evidence about the OEX. Real option prices would be needed to test the smile."])]

# ---------------------------------------------------------------- accuracy of the expansion
story += [P("How reliable is the expansion at one year?", h2),
  P("The perturbation expansion is an approximation, and its accuracy falls as maturity and vol-of-vol grow. To check it, the at-the-money implied volatility from the expansion was compared with a Monte Carlo simulation "
    "(400,000 paths, about two time steps per trading day, control variate) of the underlying model dV = κ(θ − V)dt + λV dW, correlated with spot with ρ = −0.7, using each period's fitted parameters and a flat 2% interest rate. "
    "This is a separate check written for this summary and relies on that reading of the model.", body)]
t4 = [["Period", "T (years)", "Expansion ATM IV (%)", "Monte Carlo ATM IV (%)", "Expansion minus MC (vol pts)", "MC 95% half-width (vol pts)"]]
for _, r in mc.iterrows():
    t4.append([r.Period, f"{r['T']:.2f}", f"{100*r.Expansion_ATM_IV:.2f}", f"{100*r.MC_ATM_IV:.2f}", f"{100*r.Diff:+.2f}", f"{100*r.MC_95pct_halfwidth:.2f}"])
story += [KeepTogether([P("Table 4: Expansion versus Monte Carlo, at the money, three months and one year.", cap),
          table(t4, [1.9*inch, 0.6*inch, 1.1*inch, 1.1*inch, 1.2*inch, 1.1*inch])]), Spacer(1, 6),
  P("At three months the expansion is 0.4 to 1.4 points below the simulation; at one year it is 2.9 to 9.3 points below, far outside the simulation's sampling error (under 0.1 point). "
    "The cause is likely that λ = 0.8 is large relative to the range where the expansion is accurate (λ<super>2</super>T is 0.64 at one year). Because the parameters are tuned to the expansion's own output, "
    "V<sub>0</sub>, κ and θ are effective values for this expansion, and the one-year results are the least reliable part of the surface. The 180-day tenor was not tested separately but is likely affected to a lesser degree. "
    "A lower λ, a higher-order expansion or a different pricing method (Fourier or Monte Carlo) would be needed for a trustworthy one-year surface.", body)]

# ---------------------------------------------------------------- bottom line
story += [P("Bottom line", h2),
  P("<b>Bottom line.</b> With three tuned parameters, the model reproduces the level and the slope of the OEX historical-volatility term structure out to 180 days in all regimes except the 2020 shock, "
    "including the inverted curve in 2022. Adding the 1-year maturity leaves the pre-COVID and 2022 fits close (RMSE 0.20 points each) but worsens 2023-2026 (0.72), 2021 (1.52) and 2020 (3.13), almost entirely at the new tenor. "
    "The one-year model values are also the least reliable numerically, because the expansion underestimates the model's own implied volatility by several points at that maturity. "
    "Because the market side is historical volatility and not option-implied volatility, this is a check of the model's term structure, not a validation of its smile.", body)]

# ---------------------------------------------------------------- figures
story.append(PageBreak())
story += [P("Figures", h1)]
im = Image.open(f"{OUT}/OEX_surfaces_model_vs_market.png"); w, h = im.size
story += [RLImage(f"{OUT}/OEX_surfaces_model_vs_market.png", width=W, height=W*h/w),
          P("Figure 1: Model-implied surface (coloured) versus the market historical-volatility surface (red, flat in strike), by period, for maturities up to 1 year.", cap), Spacer(1, 8)]
im2 = Image.open(f"{OUT}/OEX_term_structure_and_smiles.png"); w2, h2_ = im2.size
im2.crop((0, 0, 995, h2_)).save("fig2a.png"); im2.crop((995, 0, w2, h2_)).save("fig2b.png")
wa = 4.6*inch; wb = W
a = Image.open("fig2a.png"); b = Image.open("fig2b.png")
story += [KeepTogether([RLImage("fig2a.png", width=wa, height=wa*a.size[1]/a.size[0]),
          P("Figure 2a: Market historical-volatility term structure (solid) and model at-the-money implied volatility (dashed), 10 to 252 trading days.", cap)]),
          KeepTogether([RLImage("fig2b.png", width=wb, height=wb*b.size[1]/b.size[0]),
          P("Figure 2b: Model smile at about three months (left) and at one year (right); dotted lines are the market historical volatility.", cap)])]

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("DV", 7.5); canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(letter[0]-0.75*inch, 0.45*inch, f"{doc.page}"); canvas.restoreState()

doc = SimpleDocTemplate("/mnt/user-data/outputs/OEX_summary_section.pdf", pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch,
                        topMargin=0.7*inch, bottomMargin=0.75*inch, title="Final Output: OEX model vs market volatility surface (with 1-year maturity)", author="Yuto Sugiyama")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("ok")
