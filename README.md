# GhostSDK
The repo for Ghost Robotics SDK to work with Spirit robot.

This SDK includes all the assets from the official GhostSDK_0.18.7.zip and other custom files to run applications on Spirit robot.

This folder should be cloned and stored in the Spirit robot. As Spirit robot is not normally connected to internet, it is recommended that you also have another clone of this repo on your computer, and sync the local files/changes between your PC and Spirit using `rsync`

```
rsync -av <SOURCE> <DESTINATION>
```

Example:
```
# to pull new changes from Spirit
rsync -av ghost@192.168.168.105:/home/ghost/Desktop/DUY/SAFE/GhostSDK/ .

# to push new changes to Spirit
rsync -av . ghost@192.168.168.105:/home/ghost/Desktop/DUY/SAFE/GhostSDK/
```

Connect to Spirit's access point `Spirit40-007` with password `ghost`, user `ghost`.