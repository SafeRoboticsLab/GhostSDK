from __future__ import print_function
import numpy as np
import time
from .grmblinkpy import MBLink, Frame
#from .savePKZ import savePKZ

class MB80v2(MBLink):
    diskList = []
    diskSentCount = 0
    steppable_list = []
    steppable_sent_count = 0

    def __init__(self, sim=False, verbose=True, log=False, port=0):
        "port=0 => default RX port, otherwise use this one"
        MBLink.__init__(self) # must call this
        self.start(sim, verbose, port)
        self.rxstart()
        self.log = log
        self.floatArraySendCounter = 0
    
    def _finalizeLog(self, accum_data):
        # first turn the lists of arrays to 2D arrays
        for key in accum_data:
            accum_data[key] = np.asanyarray(accum_data[key])
        # t lumped with y for mb data
        accum_data['t'] = accum_data['y'][:,-1]
        accum_data['y'] = accum_data['y'][:-1]
        fname = savePKZ('logs/', accum_data)
        return fname
    
    def get(self):
        res = super(MB80v2, self).get() # need the super(args) for python2
        res['t'] = res['y'][-1]
        res['y'] = res['y'][:-1]
        return res

    def rxstop(self):
        accum_data = super(MB80v2, self).rxstop()
        if self.log:
            return self._finalizeLog(accum_data)

    def sendFloatArrayUpdate(self, minDecimation=10, **kwargs):
        """Send steppable, free (see BiresGaitPlanner). Manages its own state in order to stagger and sequence the sending. Call this at the control rate"""
        def prepareGoalSteppables(steppable=[], goal=[0,0,0], **kwargs):
            if len(steppable) == 0:
                return None
            assert len(goal) == 3 # must be for SE2
            goal = np.asarray(goal)
            # grab the arguments. for now assumes it does not change after this
            if len(self.steppable_list) == 0:
                self.steppable_list = steppable
            
            PATCHSZ = 9 # 9 floats to send a patch: 8*xy + z
            Ntosend = min(len(self.steppable_list) - self.steppable_sent_count, 55//PATCHSZ)
            
            # Get the range we can send this time
            step_arr = self.steppable_list[self.steppable_sent_count:self.steppable_sent_count+Ntosend]
            steppables_packed = np.hstack([np.hstack((np.reshape(s[:,:2], 8, 'C'), [np.mean(s[:,2])])) for s in step_arr])
            # create the full message
            data = np.hstack((goal, steppables_packed, np.zeros(55 - len(steppables_packed))))
            ret = [1, self.steppable_sent_count, self.steppable_sent_count + Ntosend, data]
            self.steppable_sent_count += Ntosend
            # if they have all been sent, go back to the beginning
            if self.steppable_sent_count >= len(self.steppable_list):
                self.steppable_sent_count = 0
            return ret
        
        def prepareDiskList(obsts=[], **kwargs):
            if len(obsts) == 0:
                return None
            if len(self.diskList) == 0:
                # grab the arguments. for now assumes it does not change after this
                for obst in obsts:
                    # assume each is a ConvexObstacle
                    self.diskList = self.diskList + obst.xyrs
            # TODO: sequence
            XYRSIZE = 3
            Ntosend = min(len(self.diskList) - self.diskSentCount, 58//XYRSIZE)
            # came as a list of xyr, so just concatenate them together
            diskArr = np.hstack((self.diskList[self.diskSentCount:self.diskSentCount+Ntosend]))
            data = np.hstack((diskArr, np.zeros(58 - len(diskArr))))
            ret = [2, self.diskSentCount, self.diskSentCount + Ntosend, data]
            self.diskSentCount += Ntosend
            # if they have all been sent, go back to the beginning
            if self.diskSentCount >= len(self.diskList):
                self.diskSentCount = 0
            return ret

        if self.floatArraySendCounter == 0:
            tosend = prepareGoalSteppables(**kwargs)
            if tosend is not None:
                self.sendToSim(*tosend)
        elif self.floatArraySendCounter == 5:
            tosend = prepareDiskList(**kwargs)
            if tosend is not None:
                self.sendToSim(*tosend)
        
        self.floatArraySendCounter = (self.floatArraySendCounter + 1) % minDecimation

    
    def test(self):
        self.sendBehavior(MBLink.STAND, 0)
        time.sleep(5)

        # Stand - send it once more at top of stand to enter lookaround mode, ready for WALK FIXME: send twice is correct?
        self.sendBehavior(MBLink.STAND, 0)
        time.sleep(2)

        # Enter walk mode
        self.sendBehavior(MBLink.WALK, 1)# WALK, see: http://ghostrobotics.gitlab.io/SDK/MAVLinkReference.html
        time.sleep(1)

        print(self.rxdata)

        # Walk forward
        self.sendSE2Twist([0.2, 0, 0])
        time.sleep(3)
        # Walk backwards
        self.sendSE2Twist([-0.2, 0.0, 0.0])
        time.sleep(3)
        # Turn left
        self.sendSE2Twist([0,0,0.4])
        time.sleep(3)
        # Turn right
        self.sendSE2Twist([0.0, 0.0, -0.4])
        time.sleep(3)

        # Sit
        self.sendBehavior(MBLink.SIT, 0)
        
