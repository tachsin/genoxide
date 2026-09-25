<#
.SYNOPSIS
    Pins the WSL 2 virtual machine to chosen cores, e.g. the fastest P-cores of a hybrid Intel CPU,
    so that benchmark times don't depend on which core Windows picks.

.DESCRIPTION
    Without switches, it sets the processor affinity of the WSL virtual machine (the vmmemWSL
    process) once. That needs an Administrator PowerShell and a running WSL, and lasts until the
    virtual machine stops: `wsl --shutdown`, or about a minute after the last Linux process ends.

    -Install registers a scheduled task that pins it every minute from now on and from every logon,
    in the user's session with the highest privileges, without a window, so it stays pinned. It
    logs each run to %LOCALAPPDATA%\genoxide\pin-wsl.log. -Uninstall removes the task.

    benchmarks/run.py checks the pinning before it measures times, and refuses to measure without it.

    While the file %LOCALAPPDATA%\genoxide\pin-wsl.off exists, it unpins WSL instead (every core),
    e.g. while adapters are tested in parallel; creating or deleting the file needs no admin rights.

.PARAMETER Cores
    The logical processors to pin to: 8 and 19 by default, the two favoured P-cores (the highest
    turbo frequency) of the Intel Core Ultra 7 265K the published benchmarks run on. Measured there,
    a fixed single-threaded C loop varied by about 1% on them, against 1.5% on all 8 P-cores and
    more when Windows can use the E-cores too.

.EXAMPLE
    .\pin-wsl.ps1 -Install
#>
param(
    # a list (-Cores 8,19), or one text with commas, as a scheduled task passes it
    [string[]]$Cores = @('8', '19'),
    [switch]$Install,
    [switch]$Uninstall,
    # append the outcome to %LOCALAPPDATA%\genoxide\pin-wsl.log (the scheduled task does)
    [switch]$Log
)
$ErrorActionPreference = 'Stop'
$task = 'genoxide-pin-wsl'

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    "removed the scheduled task '$task'"
    return
}

if ($Install) {
    # conhost --headless: no window every minute
    $arguments = "--headless powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Cores $($Cores -join ',') -Log"
    $action = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $arguments
    # every minute, from now on and from every logon: WSL's virtual machine restarts whenever WSL
    # starts again
    $every = (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1)).Repetition
    $now = New-ScheduledTaskTrigger -Once -At (Get-Date)
    $now.Repetition = $every
    $logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $logon.Repetition = $every
    $trigger = @($now, $logon)
    # the user's elevated session: setting the affinity of the virtual machine is refused to a
    # task that runs whether the user is logged on or not (S4U)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $task
    "installed the scheduled task '$task': it pins WSL to cores $($Cores -join ', ') at logon and every minute"
    return
}

function Write-Outcome([string]$Message) {
    if ($Log) {
        $folder = Join-Path $env:LOCALAPPDATA 'genoxide'
        New-Item -ItemType Directory -Force -Path $folder | Out-Null
        $file = Join-Path $folder 'pin-wsl.log'
        # the last 1000 lines are enough
        $lines = @(Get-Content $file -ErrorAction SilentlyContinue | Select-Object -Last 999)
        $lines + ('{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $Message) | Set-Content $file
    }
    $Message
}

$Cores = @($Cores -split ',' | ForEach-Object { [int]$_.Trim() })
$mask = [int64]0
foreach ($core in $Cores) {
    $mask = $mask -bor ([int64]1 -shl $core)
}
$where = "cores $($Cores -join ', ')"
if (Test-Path (Join-Path $env:LOCALAPPDATA 'genoxide\pin-wsl.off')) {
    $mask = ([int64]1 -shl [Environment]::ProcessorCount) - 1
    $where = 'every core (pin-wsl.off)'
}
$vm = Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vm) {
    Write-Outcome 'WSL is not running: nothing to pin'
    return
}
try {
    if ([int64]$vm.ProcessorAffinity -ne $mask) {
        $vm.ProcessorAffinity = [IntPtr]$mask
        Write-Outcome ('WSL (process {0}) pinned to {1} (mask 0x{2:X})' -f $vm.Id, $where, $mask)
    } else {
        Write-Outcome ('WSL (process {0}) already pinned to {1}' -f $vm.Id, $where)
    }
} catch {
    Write-Outcome "error: $($_.Exception.Message)"
    throw 'pinning WSL needs an Administrator PowerShell (or the scheduled task of -Install)'
}
