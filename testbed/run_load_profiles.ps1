param(
  [string]$HostUrl = "http://localhost:8080",
  [int]$NormalUsers = 20,
  [int]$StressUsers = 120,
  [string]$RunTime = "5m"
)

$ErrorActionPreference = "Stop"
$profiles = @(
  @{Name="normal"; Users=$NormalUsers; Spawn=5},
  @{Name="gradual"; Users=[int](($NormalUsers + $StressUsers) / 2); Spawn=3},
  @{Name="spike"; Users=$StressUsers; Spawn=50},
  @{Name="stress"; Users=$StressUsers; Spawn=10},
  @{Name="recovery"; Users=$NormalUsers; Spawn=10}
)

foreach ($profile in $profiles) {
  Write-Host "Running Locust profile $($profile.Name)"
  $env:LOAD_PROFILE = $profile.Name
  locust -f testbed/load/locustfile.py --headless `
    --host $HostUrl `
    --users $profile.Users `
    --spawn-rate $profile.Spawn `
    --run-time $RunTime `
    --csv "paper_artifacts/locust_$($profile.Name)"
}

