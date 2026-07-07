# run_sensitivity_weibull_k2.ps1
# EV15c: Weibull(k=2) lead-time sensitivity evaluation
# Companion to EV15 (lognormal sigma=0.5) and EV15b (sigma=0.8)
#
# Distribution: Weibull with shape k=2, scale matched to scenario mean
# Properties: lighter-tailed increasing-hazard delivery model, CV~0.52
# Models: SLA-governed supply chains (delivery more likely as wait grows)
#
# Full distribution comparison:
#   Geometric (training): memoryless, CV=1.0, P(T>2M)=13.5%
#   Lognormal sigma=0.5 (EV15):  moderate tail, P(T>2M)=5.1%
#   Lognormal sigma=0.8 (EV15b): heavy tail,    P(T>2M)=10.2%
#   Weibull k=2         (EV15c): lighter-tailed increasing-hazard model,  P(T>2M)=4.4%
#


param(
    [string[]]$Policies  = @("rlinv", "b1", "mpc"),
    [string[]]$Sites     = @("site1","site2","site3","site4","site5",
                             "site6","site7","site8","site9","site10"),
    [string[]]$Scenarios = @("normal","delayed","monsoon","extreme"),
    [int[]]$Seeds        = @(42,123,777,7,13,21,99,314,500,999),
    [int]$Episodes       = 10,
    [string]$LeadDist    = "weibull",
    [float]$LeadK        = 2.0,
    [string]$OutDir      = "results/sensitivity/weibull_k2",
    [string]$ExperimentTag = "sensitivity_weibull_k2",
    [bool]$Resume        = $true
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$readme = @"
EV15c: Weibull(k=2) lead-time sensitivity evaluation
lead_dist : weibull
lead_k    : $LeadK
OutDir    : $OutDir
Companion : results/sensitivity/lognormal/     (EV15,  sigma=0.5)
            results/sensitivity/lognormal_sigma08/ (EV15b, sigma=0.8)
Date      : $(Get-Date -Format 'yyyy-MM-dd')

Distribution properties at k=2 (Rayleigh distribution):
  Mean   : matches scenario mean exactly (by construction)
  CV     : ~0.52  (similar to lognormal sigma=0.5)
  P(T>2M): ~4.4%  (much lighter tail than geometric 13.5%)
  Hazard : increasing -- delivery more likely the longer you wait
  Models : SLA-governed supply chains

Interpretation:
  Weibull(k=2) is the OPTIMISTIC scenario -- less heavy-tailed / SLA-like increasing-hazard scenario.
  Lognormal sigma=0.8 is the PESSIMISTIC scenario -- heavy tail.
  Together they bracket the realistic space around geometric.

Do NOT mix with EV15 or EV15b results.
Report separately as EV15c.
"@
$readme | Out-File -FilePath "$OutDir/EV15c_README.txt" -Encoding utf8
Write-Host "Written: $OutDir/EV15c_README.txt" -ForegroundColor Green

$total = $Policies.Count * $Sites.Count * $Scenarios.Count * $Seeds.Count
$count = 0
$failed = @()

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EV15c: Weibull(k=2) Sensitivity" -ForegroundColor Cyan
Write-Host "  LeadDist: $LeadDist (k=$LeadK)" -ForegroundColor Cyan
Write-Host "  OutDir  : $OutDir" -ForegroundColor Cyan
Write-Host "  Total   : $total runs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
$KLabel = ("k" + ($LeadK.ToString("0.0") -replace "\.","p"))
foreach ($policy in $Policies) {
    foreach ($site in $Sites) {
        foreach ($scenario in $Scenarios) {
            foreach ($seed in $Seeds) {
                $count++
                #$outFile = "$OutDir/${policy}_${site}_${scenario}_weibull_k2_s${seed}.csv"
                $outFile = "$OutDir/${policy}_${site}_${scenario}_weibull_${KLabel}_s${seed}.csv"

                if ($Resume -and (Test-Path $outFile)) { continue }

                Write-Host "[$count/$total] $policy / $site / $scenario / seed=$seed" -NoNewline

                try {
                    if ($policy -eq "rlinv") {
                        $modelPath = "runs/rlinv/${site}_s${seed}_final.zip"
                        if (-not (Test-Path $modelPath)) {
                            Write-Host " [SKIP - no model]" -ForegroundColor Yellow
                            $failed += "$policy/$site/$scenario/seed=$seed"
                            continue
                        }
                        python src/eval/evaluate.py `
                            --site $site --lead $scenario `
                            --model_path $modelPath `
                            --policy_type rl --algo maskable `
                            --episodes $Episodes --seed $seed `
                            --episode_len 360 `
                            --lead_dist $LeadDist --lead_k $LeadK `
                            --policy_label "rlinv_weibull_$KLabel" `
                            --train_scenario normal `
                            --experiment_tag $ExperimentTag `
                            --out_csv $outFile
                    } elseif ($policy -eq "b1") {
                        python src/eval/evaluate.py `
                            --site $site --lead $scenario `
                            --policy_type b1 `
                            --episodes $Episodes --seed $seed `
                            --episode_len 360 `
                            --lead_dist $LeadDist --lead_k $LeadK `
                            --policy_label "b1_weibull_$KLabel" `
                            --train_scenario normal `
                            --experiment_tag $ExperimentTag `
                            --out_csv $outFile
                    } else {
                        python src/eval/evaluate.py `
                            --site $site --lead $scenario `
                            --policy_type mpc `
                            --episodes $Episodes --seed $seed `
                            --episode_len 360 `
                            --lead_dist $LeadDist --lead_k $LeadK `
                            --policy_label "mpc_weibull_$KLabel" `
                            --train_scenario normal `
                            --experiment_tag $ExperimentTag `
                            --out_csv $outFile
                    }

                    if ($LASTEXITCODE -eq 0) {
                        Write-Host " OK" -ForegroundColor Green
                    } else {
                        Write-Host " FAILED" -ForegroundColor Red
                        $failed += "$policy/$site/$scenario/seed=$seed"
                    }
                } catch {
                    Write-Host " ERROR: $_" -ForegroundColor Red
                    $failed += "$policy/$site/$scenario/seed=$seed"
                }
            }
        }
    }
}

Write-Host ""
Write-Host "Done: $count/$total, failed: $($failed.Count)" -ForegroundColor Cyan
if ($failed.Count -gt 0) {
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}
Write-Host "Next: python analyse_lognormal_sensitivity.py --input_dir $OutDir --label weibull_k2" -ForegroundColor Yellow
