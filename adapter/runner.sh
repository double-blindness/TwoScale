#!/bin/bash

# Start the server
python main.py &
  
# Start the signal
python adaptation_signal.py &

handle_sigterm() {
  echo "Received SIGTERM. Exiting..."

  exit 0
}

trap handle_sigterm SIGTERM

# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?