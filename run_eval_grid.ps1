# run_eval_grid.ps1 — Full Phase 3 evaluation grid
#
# Evaluates all trained policies across all sites, lead scenarios, and seeds.
# Writes one CSV per (policy, site, lead, seed) to results/phase3/<policy>/.
#
# Usage:
#   # Quick sanity (site5 only, seed42 only)
#   .\run_eval_grid.ps1 -Sites site5 -Seeds 42 -Resume
#
#   # Full grid — all policies, all sites, all seeds (~5-6 hours)
#   .\run_eval_grid.ps1 -Resume
#
#   # Single policy
#   .\run_eval_grid.ps1 -Policies rlinv,b1 -Resume
#
# Output naming convention (consistent with build_master_summary.py regex):
#   results/phase3/<policy>/<policy>_<site>_<lead>_s<seed>.csv

param(
    [string]$Sites    = "all",
    [string]$Policies = "all",
    [string]$Leads    = "all",
    [string]$Seeds    = "all",
    [int]$Episodes    = 10,
    [switch]$Resume
)

$ErrorActionPreference = "Continue"

# ── Site list ──────────────────────────────────────────────────────────────────
if ($Sites -eq "all") {
    $siteList = 1..10 | ForEach-Object { "site$_" }
} else {
    $siteList = $Sites -split ","
}

# ── Lead scenarios ─────────────────────────────────────────────────────────────
if ($Leads -eq "all") {
    $leadList = @("normal", "delayed", "monsoon", "extreme")
} else {
    $leadList = $Leads -split ","
}

# ── Seeds ─────────────────────────────────────────────────────────────────────
if ($Seeds -eq "all") {
    $seedList = @(42, 123, 777, 7, 13, 21, 99, 314, 500, 999)
} else {
    $seedList = $Seeds -split "," | ForEach-Object { [int]$_ }
}

# ── Policy configuration ───────────────────────────────────────────────────────
# ModelDir   : where trained models live (empty for baselines)
# Algo       : maskable or ppo (empty for baselines)
# EnvType    : --env_type flag value (empty = use default track_a / standard env)
# TrainScen  : --train_scenario metadata tag
# TrainSteps : --train_steps metadata tag
# IsBaseline : no model file, use --policy_type directly

$AllPolicies = [ordered]@{
    "rlinv"  = @{ ModelDir="runs/rlinv";         Algo="maskable"; EnvType="";        TrainScen="normal"; TrainSteps=400000; IsBaseline=$false }
    "multi"  = @{ ModelDir="runs/rlinv_multi";   Algo="maskable"; EnvType="";        TrainScen="multi";  TrainSteps=500000; IsBaseline=$false }
    "trackb" = @{ ModelDir="runs/trackb";         Algo="ppo";      EnvType="track_b"; TrainScen="normal"; TrainSteps=400000; IsBaseline=$false }
    "a5"     = @{ ModelDir="runs/ablation_a5";    Algo="maskable"; EnvType="a5";      TrainScen="normal"; TrainSteps=400000; IsBaseline=$false }
    "a6"     = @{ ModelDir="runs/ablation_a6";    Algo="maskable"; EnvType="a6";      TrainScen="normal"; TrainSteps=400000; IsBaseline=$false }
    "a7"     = @{ ModelDir="runs/ablation_a7";    Algo="ppo";      EnvType="";        TrainScen="normal"; TrainSteps=400000; IsBaseline=$false }
    "b0"     = @{ ModelDir="";                    Algo="";         EnvType="";        TrainScen="normal";    TrainSteps=0;      IsBaseline=$true }
    "b1"     = @{ ModelDir="";                    Algo="";         EnvType="";        TrainScen="normal";    TrainSteps=0;      IsBaseline=$true }
}

if ($Policies -eq "all") {
    $policyList = $AllPolicies.Keys
} else {
    $policyList = $Policies -split ","
}

# ── Counters ───────────────────────────────────────────────────────────────────
$completed = 0
$skipped   = 0
$failed    = @()

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 3 Full Evaluation Grid" -ForegroundColor Cyan
Write-Host "  Policies : $($policyList -join ', ')" -ForegroundColor Cyan
Write-Host "  Sites    : $($siteList -join ', ')" -ForegroundColor Cyan
Write-Host "  Leads    : $($leadList -join ', ')" -ForegroundColor Cyan
Write-Host "  Seeds    : $($seedList -join ', ')" -ForegroundColor Cyan
Write-Host "  Episodes : $Episodes" -ForegroundColor Cyan
Write-Host "  Resume   : $Resume" -ForegroundColor Cyan
Write-Host "  Total    : $($policyList.Count * $siteList.Count * $leadList.Count * $seedList.Count) evals" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Main eval loop ─────────────────────────────────────────────────────────────
foreach ($policy in $policyList) {
    if (-not $AllPolicies.Contains($policy)) {
        Write-Host "[ERROR] Unknown policy: $policy" -ForegroundColor Red
        continue
    }

    $cfg    = $AllPolicies[$policy]
    $outDir = "results/phase3/$policy"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    foreach ($site in $siteList) {
        foreach ($lead in $leadList) {
            foreach ($seed in $seedList) {

                $outCsv = "$outDir/${policy}_${site}_${lead}_s${seed}.csv"

                # ── Resume: skip if output already exists ──────────────────
                if ($Resume -and (Test-Path $outCsv)) {
                    $skipped++
                    continue
                }

                Write-Host "=== $policy | $site | $lead | seed=$seed ===" -ForegroundColor Cyan

                try {
                    if ($cfg.IsBaseline) {
                        # ── Baseline (B0 / B1) — no model file ────────────
                        python src/eval/evaluate.py `
                            --site $site --lead $lead `
                            --policy_type $policy `
                            --episodes $Episodes --seed $seed `
                            --episode_len 360 `
                            --policy_label $policy `
                            --train_scenario $($cfg.TrainScen) `
                            --experiment_tag "phase3_$policy" `
                            --out_csv $outCsv

                    } else {
                        # ── RL policy — requires model file ───────────────
                        $modelPath = "$($cfg.ModelDir)/${site}_s${seed}_final.zip"

                        if (-not (Test-Path $modelPath)) {
                            Write-Host "  [SKIP] Model not found: $modelPath" -ForegroundColor Yellow
                            continue
                        }

                        if ($cfg.EnvType -ne "") {
                            python src/eval/evaluate.py `
                                --site $site --lead $lead `
                                --model_path $modelPath `
                                --policy_type rl --algo $($cfg.Algo) `
                                --env_type $($cfg.EnvType) `
                                --episodes $Episodes --seed $seed `
                                --episode_len 360 `
                                --policy_label $policy `
                                --train_scenario $($cfg.TrainScen) `
                                --train_steps $($cfg.TrainSteps) `
                                --experiment_tag "phase3_$policy" `
                                --out_csv $outCsv
                        } else {
                            python src/eval/evaluate.py `
                                --site $site --lead $lead `
                                --model_path $modelPath `
                                --policy_type rl --algo $($cfg.Algo) `
                                --episodes $Episodes --seed $seed `
                                --episode_len 360 `
                                --policy_label $policy `
                                --train_scenario $($cfg.TrainScen) `
                                --train_steps $($cfg.TrainSteps) `
                                --experiment_tag "phase3_$policy" `
                                --out_csv $outCsv
                        }
                    }

                    if ($LASTEXITCODE -ne 0) {
                        throw "Python exited with code $LASTEXITCODE"
                    }

                    $completed++

                } catch {
                    Write-Host "FAILED: $policy | $site | $lead | seed=$seed" -ForegroundColor Red
                    Write-Host $_.Exception.Message -ForegroundColor Red
                    $failed += "$policy,$site,$lead,$seed"
                }
            }
        }
    }
}

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Eval Grid Complete" -ForegroundColor Green
Write-Host "  Completed : $completed" -ForegroundColor Green
Write-Host "  Skipped   : $skipped" -ForegroundColor Yellow

if ($failed.Count -gt 0) {
    $failed | Out-File "results/failed_eval_runs.txt"
    Write-Host "  Failed    : $($failed.Count) - see results/failed_eval_runs.txt" -ForegroundColor Red
} else {
    Write-Host "  Failed    : 0" -ForegroundColor Green
}

Write-Host "========================================" -ForegroundColor Green
