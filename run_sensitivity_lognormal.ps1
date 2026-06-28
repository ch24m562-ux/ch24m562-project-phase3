# run_sensitivity_lognormal.ps1
# Lognormal lead-time sensitivity evaluation.
# Part of the sensitivity experiment family:
#   results/sensitivity/lognormal/  (this script)
#   results/sensitivity/gamma/      (future)
#   results/sensitivity/reward/     (future)
#
# Purpose: test whether the geometric->lognormal distributional change
# affects conclusions. This is a post-hoc sensitivity analysis and is
# NOT part of the canonical evaluation grid used to establish the
# principal thesis results (results/phase3/master_summary.csv).
# Eval-only -- no retraining. Same models, same seeds, same episodes.
# Only delivery timing distribution changes. Mean delivery time is
# IDENTICAL (matched via mu = ln(mean) - sigma^2/2).
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\run_sensitivity_lognormal.ps1

param(
    [string[]]$Policies  = @("rlinv", "b1", "mpc"),
    [string[]]$Sites     = @("site1","site2","site3","site4","site5",
                             "site6","site7","site8","site9","site10"),
    [string[]]$Scenarios = @("normal","delayed","monsoon","extreme"),
    [int[]]$Seeds        = @(42,123,777,7,13,21,99,314,500,999),
    [int]$Episodes       = 10,
    [string]$LeadDist    = "lognormal",
    [float]$LeadSigma    = 0.5,
    [string]$OutDir      = "results/sensitivity/lognormal",
    [bool]$Resume        = $true
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$total   = $Policies.Count * $Sites.Count * $Scenarios.Count * $Seeds.Count
$count   = 0
$skipped = 0
$failed  = @()

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lognormal Lead-Time Sensitivity Eval" -ForegroundColor Cyan
Write-Host "  Policies : $($Policies -join ', ')" -ForegroundColor Cyan
Write-Host "  Scenarios: $($Scenarios -join ', ')" -ForegroundColor Cyan
Write-Host "  Seeds    : $($Seeds -join ', ')" -ForegroundColor Cyan
Write-Host "  LeadDist : $LeadDist (sigma=$LeadSigma)" -ForegroundColor Cyan
Write-Host "  OutDir   : $OutDir" -ForegroundColor Cyan
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

                # Match run_eval_grid.ps1 filename convention: _s{seed}
                $outFile = "$OutDir/${policy}_${site}_${scenario}_lognormal_s${seed}.csv"

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
                            --site         $site `
                            --lead         $scenario `
                            --model_path   $modelPath `
                            --policy_type  rl `
                            --algo         maskable `
                            --episodes     $Episodes `
                            --seed         $seed `
                            --episode_len  360 `
                            --lead_dist    $LeadDist `
                            --policy_label "rlinv_lognormal" `
                            --train_scenario normal `
                            --train_steps  400000 `
                            --experiment_tag "sensitivity_lognormal" `
                            --out_csv      $outFile
                    }
                    elseif ($policy -eq "b1") {
                        python src/eval/evaluate.py `
                            --site         $site `
                            --lead         $scenario `
                            --policy_type  b1 `
                            --episodes     $Episodes `
                            --seed         $seed `
                            --episode_len  360 `
                            --lead_dist    $LeadDist `
                            --policy_label "b1_lognormal" `
                            --train_scenario normal `
                            --experiment_tag "sensitivity_lognormal" `
                            --out_csv      $outFile
                    }
                    elseif ($policy -eq "mpc") {
                        python src/eval/evaluate.py `
                            --site         $site `
                            --lead         $scenario `
                            --policy_type  mpc `
                            --episodes     $Episodes `
                            --seed         $seed `
                            --episode_len  360 `
                            --lead_dist    $LeadDist `
                            --policy_label "mpc_lognormal" `
                            --train_scenario normal `
                            --experiment_tag "sensitivity_lognormal" `
                            --out_csv      $outFile
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
Write-Host "  python analyse_lognormal_sensitivity.py" -ForegroundColor Yellow
Write-Host "  -> produces results/sensitivity/lognormal/lognormal_summary.csv" -ForegroundColor Yellow
Write-Host "  -> produces results/sensitivity/lognormal/lognormal_thesis_summary.txt" -ForegroundColor Yellow
