# run_sensitivity_lognormal_sigma08.ps1
# EV15b: Heavy-tail lognormal sensitivity evaluation (sigma=0.8)
# Companion to run_sensitivity_lognormal.ps1 (sigma=0.5 / EV15)
#
# Key differences from EV15:
#   LeadSigma  : 0.8  (vs 0.5 in EV15) -- CV~0.90, heavier tail
#   OutDir     : results/sensitivity/lognormal_sigma08/
#   PolicyLabels: rlinv_lognormal08 / b1_lognormal08 / mpc_lognormal08
#   ExperimentTag: sensitivity_lognormal_sigma08
#
# Distribution properties at sigma=0.8 (matched mean):
#   P(delivery > 2x mean) ~ 10.2% (vs 5.1% at sigma=0.5)
#   99th percentile: ~4.8x mean (vs ~2.9x at sigma=0.5)
#
# Requires: --lead_sigma patch applied to src/eval/evaluate.py
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\run_sensitivity_lognormal_sigma08.ps1

param(
    [string[]]$Policies  = @("rlinv", "b1", "mpc"),
    [string[]]$Sites     = @("site1","site2","site3","site4","site5",
                             "site6","site7","site8","site9","site10"),
    [string[]]$Scenarios = @("normal","delayed","monsoon","extreme"),
    [int[]]$Seeds        = @(42,123,777,7,13,21,99,314,500,999),
    [int]$Episodes       = 10,
    [string]$LeadDist    = "lognormal",
    [float]$LeadSigma    = 0.8,
    [string]$OutDir      = "results/sensitivity/lognormal_sigma08",
    [string]$ExperimentTag = "sensitivity_lognormal_sigma08",
    [bool]$Resume        = $true
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Write README so future-you knows what this folder contains
$readme = @"
EV15b: Heavy-tail lognormal sensitivity evaluation
sigma     : $LeadSigma
OutDir    : $OutDir
Companion : results/sensitivity/lognormal/ (EV15, sigma=0.5)
Policies  : $($Policies -join ', ')
Sites     : all 10
Seeds     : $($Seeds -join ', ')
Episodes  : $Episodes
Date      : $(Get-Date -Format 'yyyy-MM-dd')

Distribution properties vs EV15:
  sigma=0.5 (EV15):  P(delivery>2x mean)~5%,  99th pct ~2.9x mean
  sigma=0.8 (EV15b): P(delivery>2x mean)~10%, 99th pct ~4.8x mean
  Means are MATCHED across scenarios (same as EV15).

Do NOT mix these files with EV15 (lognormal/) results.
Report separately: EV15 = moderate variability, EV15b = heavy tail.
"@
$readme | Out-File -FilePath "$OutDir/EV15b_README.txt" -Encoding utf8
Write-Host "Written: $OutDir/EV15b_README.txt" -ForegroundColor Green

$total   = $Policies.Count * $Sites.Count * $Scenarios.Count * $Seeds.Count
$count   = 0
$skipped = 0
$failed  = @()

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EV15b: Lognormal Sensitivity (sigma=0.8)" -ForegroundColor Cyan
Write-Host "  Policies : $($Policies -join ', ')" -ForegroundColor Cyan
Write-Host "  Scenarios: $($Scenarios -join ', ')" -ForegroundColor Cyan
Write-Host "  Seeds    : $($Seeds -join ', ')" -ForegroundColor Cyan
Write-Host "  LeadDist : $LeadDist (sigma=$LeadSigma)" -ForegroundColor Cyan
Write-Host "  OutDir   : $OutDir" -ForegroundColor Cyan
Write-Host "  Tag      : $ExperimentTag" -ForegroundColor Cyan
Write-Host "  Total    : $total runs" -ForegroundColor Cyan
Write-Host "  Note     : output is OUTSIDE results/phase3/ so" -ForegroundColor Yellow
Write-Host "             build_master_summary.py will NOT pick it up." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($policy in $Policies) {
    foreach ($site in $Sites) {
        foreach ($scenario in $Scenarios) {
            foreach ($seed in $Seeds) {
                $count++

                # Filename convention matches EV15 but with _sigma08 suffix
                $outFile = "$OutDir/${policy}_${site}_${scenario}_lognormal08_s${seed}.csv"

                if ($Resume -and (Test-Path $outFile)) {
                    $skipped++
                    continue
                }

                Write-Host "[$count/$total] $policy / $site / $scenario / seed=$seed" -NoNewline

                try {
                    if ($policy -eq "rlinv") {
                        $modelPath = "runs/rlinv/${site}_s${seed}_final.zip"
                        if (-not (Test-Path $modelPath)) {
                            Write-Host " [SKIP - model not found: $modelPath]" -ForegroundColor Yellow
                            $failed += "$policy/$site/$scenario/seed=$seed (no model)"
                            continue
                        }
                        python src/eval/evaluate.py `
                            --site           $site `
                            --lead           $scenario `
                            --model_path     $modelPath `
                            --policy_type    rl `
                            --algo           maskable `
                            --episodes       $Episodes `
                            --seed           $seed `
                            --episode_len    360 `
                            --lead_dist      $LeadDist `
                            --lead_sigma     $LeadSigma `
                            --policy_label   "rlinv_lognormal08" `
                            --train_scenario normal `
                            --train_steps    400000 `
                            --experiment_tag $ExperimentTag `
                            --out_csv        $outFile
                    }
                    elseif ($policy -eq "b1") {
                        python src/eval/evaluate.py `
                            --site           $site `
                            --lead           $scenario `
                            --policy_type    b1 `
                            --episodes       $Episodes `
                            --seed           $seed `
                            --episode_len    360 `
                            --lead_dist      $LeadDist `
                            --lead_sigma     $LeadSigma `
                            --policy_label   "b1_lognormal08" `
                            --train_scenario normal `
                            --experiment_tag $ExperimentTag `
                            --out_csv        $outFile
                    }
                    elseif ($policy -eq "mpc") {
                        python src/eval/evaluate.py `
                            --site           $site `
                            --lead           $scenario `
                            --policy_type    mpc `
                            --episodes       $Episodes `
                            --seed           $seed `
                            --episode_len    360 `
                            --lead_dist      $LeadDist `
                            --lead_sigma     $LeadSigma `
                            --policy_label   "mpc_lognormal08" `
                            --train_scenario normal `
                            --experiment_tag $ExperimentTag `
                            --out_csv        $outFile
                    }

                    if ($LASTEXITCODE -eq 0) {
                        Write-Host " OK" -ForegroundColor Green
                    } else {
                        Write-Host " FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
                        $failed += "$policy/$site/$scenario/seed=$seed"
                    }
                }
                catch {
                    Write-Host " ERROR: $_" -ForegroundColor Red
                    $failed += "$policy/$site/$scenario/seed=$seed"
                }
            }
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done: $count/$total runs, $skipped skipped" -ForegroundColor Cyan
if ($failed.Count -gt 0) {
    Write-Host "FAILED ($($failed.Count)):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
} else {
    Write-Host "All completed successfully." -ForegroundColor Green
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  python analyse_lognormal_sensitivity.py --input_dir results/sensitivity/lognormal_sigma08 --sigma 0.8" -ForegroundColor Yellow
Write-Host "  -> produces results/sensitivity/lognormal_sigma08/lognormal08_summary.csv" -ForegroundColor Yellow
