$ErrorActionPreference = "Stop"
$repo = "rayenboussayed/autoclip"
$version = if ($env:AUTOCLIP_VERSION) { $env:AUTOCLIP_VERSION } else { "latest" }
$homeDir = if ($env:AUTOCLIP_HOME) { $env:AUTOCLIP_HOME } else { Join-Path $env:LOCALAPPDATA "autoclip" }
$asset = "autoclip-windows-x64.zip"
$base = if ($env:AUTOCLIP_BASE_URL) { $env:AUTOCLIP_BASE_URL }
        elseif ($version -eq "latest") { "https://github.com/$repo/releases/latest/download" }
        else { "https://github.com/$repo/releases/download/v$($version.TrimStart('v'))" }
$tmp = New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ("autoclip-" + [guid]::NewGuid()))
try {
  Invoke-WebRequest "$base/$asset" -OutFile "$tmp\$asset" -UseBasicParsing
  Invoke-WebRequest "$base/SHA256SUMS" -OutFile "$tmp\SHA256SUMS" -UseBasicParsing
  $line = Get-Content "$tmp\SHA256SUMS" | Where-Object { $_ -match [regex]::Escape($asset) } | Select-Object -First 1
  if (-not $line) { throw "no checksum listed for $asset" }
  $expected = ($line -split '\s+')[0].ToLower()
  $actual = (Get-FileHash "$tmp\$asset" -Algorithm SHA256).Hash.ToLower()
  if ($expected -ne $actual) { throw "checksum mismatch for $asset" }
  $app = Join-Path $homeDir "app"
  New-Item -ItemType Directory -Force $homeDir | Out-Null
  if (Test-Path $app) { Remove-Item $app -Recurse -Force }
  Expand-Archive "$tmp\$asset" -DestinationPath $homeDir -Force
  Rename-Item (Join-Path $homeDir "autoclip") $app
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if ($userPath -notlike "*$app*") { [Environment]::SetEnvironmentVariable("Path", "$userPath;$app", "User") }
  Write-Host "Installed: $app\autoclip.exe (open a new terminal)"
} finally { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }