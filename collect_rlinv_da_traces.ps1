# collect_rlinv_da_traces.ps1
# Collects 72 trace files (1 episode each) for population-level behaviour analysis
# Runtime: ~2-3 minutes

param(
    [string[]]$Sites     = @("site2","site5","site7"),
    [string[]]$Seeds     = @(42,123,777),
    [string[]]$Scenarios = @("normal","delayed","monsoon","extreme"),
    [string]$TraceDir    = "results/traces/rlinv_da/all"
)

New-Item -ItemType Directory -Force -Path $TraceDir | Out-Null
$total = 2 * $Sites.Count * $Seeds.Count * $Scenarios.Count
$count = 0

Write-Host "Collecting $total traces for behaviour analysis..." -ForegroundColor Cyan

foreach ($site in $Sites) {
    foreach ($seed in $Seeds) {
        foreach ($sc in $Scenarios) {
            $count++

            # Base-regime trace
            $out = "$TraceDir/base_${site}_${sc}_s${seed}_ep0.npz"
            if (-not (Test-Path $out)) {
                Write-Host "[$count/$total] base $site/$sc/s$seed" -NoNewline
                python src/eval/evaluate.py `
                    --site $site --lead $sc --seed $seed `
                    --episodes 1 --episode_len 360 `
                    --policy_type rl --algo maskable `
                    --model_path "runs/rlinv_da/base_regime/${site}_s${seed}_final.zip" `
                    --train_scenario normal `
                    --lead_dist lognormal --lead_sigma 0.5 `
                    --use_supplier_regime `
                    --policy_label rlinv_base_regime `
                    --trace_out $out `
                    --out_csv "$TraceDir/base_${site}_${sc}_s${seed}.csv"
                Write-Host " OK" -ForegroundColor Green
            }

            $count++

            # EWMA-regime trace
            $out = "$TraceDir/ewma_${site}_${sc}_s${seed}_ep0.npz"
            if (-not (Test-Path $out)) {
                Write-Host "[$count/$total] ewma $site/$sc/s$seed" -NoNewline
                python src/eval/evaluate.py `
                    --site $site --lead $sc --seed $seed `
                    --episodes 1 --episode_len 360 `
                    --policy_type rl --algo maskable `
                    --model_path "runs/rlinv_da/ewma_regime/${site}_s${seed}_final.zip" `
                    --train_scenario normal `
                    --lead_dist lognormal --lead_sigma 0.5 `
                    --use_supplier_regime --use_ewma_lead `
                    --policy_label rlinv_da_ewma `
                    --trace_out $out `
                    --out_csv "$TraceDir/ewma_${site}_${sc}_s${seed}.csv"
                Write-Host " OK" -ForegroundColor Green
            }
        }
    }
}
Write-Host "Done. Run: python analyse_rlinv_da_behaviour.py" -ForegroundColor Yellow
