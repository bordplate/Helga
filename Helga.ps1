param (
    [string]$Environment,
    [string]$RedisHost = "localhost",
    [int]$RedisPort = 6379,
    [string]$Model,
    [bool]$Wandb = $false,
    [bool]$Commit,
    [int]$NumWorkers = 1,
    [string]$ProjectKey = "rac1.fitness-course"
)

# Function to start a Python process
function Start-ProcessWithArguments {
    param (
        [bool]$Learner,
        [string]$Arguments,
        [bool]$EvalFlag = $false
    )

    # If learner flag is set, we set $Program to 'learner.py', otherwise 'worker.py'
    if ($Learner) {
        $Program = "learner.py"
    } else {
        $Program = "worker.py"
    }

    $Command = "cd agent;python $Program $Arguments" # Replace with actual Python command
    if ($EvalFlag) {
        $Command = $Command + " --eval"
    }

    $Process = Start-Process PowerShell -ArgumentList "-NoExit", "-Command", $Command -PassThru
    return $Process
}

# Initialize commit based on environment
$Commit = -not [System.Diagnostics.Debugger]::IsAttached

# Construct base arguments
$BaseArgs = "--environment $Environment --redis-host $RedisHost --redis-port $RedisPort"
if ($Model) {
    $BaseArgs += " --model $Model"
}
$BaseArgs += " --wandb:$Wandb --commit:$Commit --project-key $ProjectKey"

# Start learner process
$LearnerProcess = Start-ProcessWithArguments -Arguments $BaseArgs

# Start worker processes
$WorkerProcesses = @()
1..$NumWorkers | ForEach-Object {
    $WorkerProcesses += Start-ProcessWithArguments -Arguments $BaseArgs
}

# Start evaluation worker
$EvalProcess = Start-ProcessWithArguments -Arguments $BaseArgs -EvalFlag $true

# Handle process restarting based on user input
Write-Host "Enter `1` to restart learner, `2` to restart evaluation worker, `3` to restart other workers:"
while ($true) {
    $Input = Read-Host "Please enter a command"
    switch ($Input) {
        '1' {
            $LearnerProcess | Stop-Process -Force
            $LearnerProcess = Start-ProcessWithArguments -Arguments $BaseArgs
            Write-Host "Learner restarted."
        }
        '2' {
            $EvalProcess | Stop-Process -Force
            $EvalProcess = Start-ProcessWithArguments -Arguments $BaseArgs -EvalFlag $true
            Write-Host "Evaluation worker restarted."
        }
        '3' {
            $WorkerProcesses | Stop-Process -Force
            $WorkerProcesses = @()
            1..$NumWorkers | ForEach-Object {
                $WorkerProcesses += Start-ProcessWithArguments -Arguments $BaseArgs
            }
            Write-Host "Workers restarted."
        }
        default {
            Write-Host "Invalid input. Use `1`, `2`, or `3`."
        }
    }
}