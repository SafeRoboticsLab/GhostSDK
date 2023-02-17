if [ "$1" == "" ]; then
    echo "Failed. Usage => upload_ota.sh [binfile]"
elif [ "$1" != "main.bin" ]; then
    echo "Rename the binary to main.bin and try again. You can use mv <source> <dest>"
else
    echo "Uploading binary..."
    scp main.bin ghost@192.168.168.105:~/current_ros2/share/ghost_manager/scripts/
    echo "Flashing..."
    ssh ghost@192.168.168.105 "date && cd ~/current_ros2/share/ghost_manager/scripts/ && pwd && ls -l && chmod +x reset_mainboard.sh updateMainboard.sh upload.sh && ./updateMainboard.sh"
fi
