# run_mpc_grid.ps1
# Drives evaluate.py across the full site x scenario x seed grid for a given
# policy_type, since evaluate.py itself only accepts ONE --site, ONE --lead,
# and ONE --seed per invocation (confirmed from its argparse usage string --
# there is no --site all / --lead all / --seeds batch mode).
#
# Each invocation writes its own CSV (df.to_csv overwrites, no append mode in
# evaluate.py), so this script writes one file per (site, lead, seed) combo
# into a per-policy subfolder, to be merged afterward by build_master_summary.py.
#
# USAGE:
#   .\run_mpc_grid.ps1 -PolicyType mpc -PolicyLabel mpc
#   .\run_mpc_grid.ps1 -PolicyType mpc -PolicyLabel mpc_forecast -ForecastCache results/forecasts/forecast_cache_H24.pkl
#   .\run_mpc_grid.ps1 -PolicyType oracle -PolicyLabel oracle_mpc -Episodes 5 -Seeds 42,123,777

param(
    [Parameter(Mandatory=$true)][string]$PolicyType,
    [Parameter(Mandatory=$true)][string]$PolicyLabel,
    [string]$ForecastCache = "",
    [int]$MpcHorizon = 24,
    [int]$Episodes = 10,
    [int[]]$Seeds = @(42, 123, 777, 1, 2, 3, 4, 5, 6, 7),
    [string[]]$Sites = @("site1","site2","site3","site4","site5","site6","site7","site8","site9","site10"),
    [string[]]$Scenarios = @("normal","delayed","monsoon","extreme"),
    [string]$OutDir = "results/phase3"
)

$ErrorActionPreference = "Stop"

# build_master_summary.py (no CLI args, hardcoded to rglob results/phase3/*.csv)
# requires filenames matching EXACTLY:
#   ^(?P<policy>[a-zA-Z0-9_]+)_(?P<site>site\d+)_(?P<lead>normal|delayed|monsoon|extreme|very_delayed|fast|no_delay|multi)(?:_(?P<tag>[^.]+))?\.csv$
# i.e. policy_site_lead[_tag].csv -- e.g. mpc_site5_normal_seed42.csv
# Getting this wrong means parse_filename() silently returns empty metadata
# and the run becomes unattributable in master_summary.csv.
$policyDir = Join-Path $OutDir $PolicyLabel
New-Item -ItemType Directory -Force -Path $policyDir | Out-Null

$total = $Sites.Count * $Scenarios.Count * $Seeds.Count
$count = 0
$failed = @()

Write-Host "=== Running $PolicyLabel grid: $($Sites.Count) sites x $($Scenarios.Count) scenarios x $($Seeds.Count) seeds = $total runs ==="

foreach ($site in $Sites) {
    foreach ($scenario in $Scenarios) {
        foreach ($seed in $Seeds) {
            $count++
            $outFile = Join-Path $policyDir "$($PolicyLabel)_$($site)_$($scenario)_seed$($seed).csv"

            # Skip if already done (resume support) -- delete the file manually
            # to force a rerun of a specific combo.
            if (Test-Path $outFile) {
                Write-Host "[$count/$total] SKIP (exists): $site / $scenario / seed=$seed"
                continue
            }

            Write-Host "[$count/$total] $site / $scenario / seed=$seed"

            $argList = @(
                "src/eval/evaluate.py",
                "--site", $site,
                "--lead", $scenario,
                "--policy_type", $PolicyType,
                "--mpc_horizon", $MpcHorizon,
                "--episodes", $Episodes,
                "--seed", $seed,
                "--out_csv", $outFile,
                "--policy_label", $PolicyLabel
            )
            if ($ForecastCache -ne "") {
                $argList += @("--forecast_cache", $ForecastCache)
            }

            python @argList
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  !! FAILED: $site / $scenario / seed=$seed (exit code $LASTEXITCODE)" -ForegroundColor Red
                $failed += "$site/$scenario/seed=$seed"
            }
        }
    }
}

Write-Host ""
Write-Host "=== Done: $count/$total runs attempted ==="
if ($failed.Count -gt 0) {
    Write-Host "FAILED runs ($($failed.Count)):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
} else {
    Write-Host "All runs completed successfully." -ForegroundColor Green
}
Write-Host "Per-run CSVs in: $policyDir"
Write-Host "Next: merge with build_master_summary.py"
