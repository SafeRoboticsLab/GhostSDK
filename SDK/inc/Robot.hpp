/*
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include <queue>
#include <atomic>
#include <vector>
#include <stdint.h>
#include "SMath.h"
#include "Behavior.h"
#include "simple.pb.h"
#ifdef _MSC_VER
#pragma warning(disable : 4244)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <mavlink/common/mavlink.h>
#ifdef _MSC_VER
#pragma warning(default : 4244)
#else
#pragma GCC diagnostic pop
#endif

namespace gr {

/** @addtogroup Robot Robot
 * @brief Access various robot functionality
 *  @{
 */

// TODO: reinstate non-fixed size
typedef Eigen::Matrix<float, 3, 3> MatMN_t; // Keep this type regular column major
typedef Eigen::Matrix<float, 3, 1> VecN_t;
typedef Eigen::Matrix<float, 3, 1> VecM_t;
// For converting from float arrays
typedef Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > MCMatMN;
typedef Eigen::Map<VecM_t> MVecM;
typedef Eigen::Map<VecN_t> MVecN;
typedef Eigen::Map<const VecN_t> MCVecN;
typedef Eigen::Map<const VecM_t> MCVecM;

// Order of the joints in the limb
#define HIPJ (0)
#define KNEEJ (1)
#define ABDUCTIONJ (2)

/**
 * @brief Robot class with methods to access sensor data, actuate motors, and control the robot behavior
 */
class Robot : public RobotData {
public:
	const static uint32_t ACT_SIT = 0;
	const static uint32_t ACT_STAND = 1;
	const static uint32_t ACT_WALK = 2;
	/**
	 * @brief Initialize the SDK for a particular robot platform
	 * 
	 * @param type Robot type (see RobotParams::Type)
	 * @param argc Number of command line arguments (pass from main())
	 * @param argv Command line arguments (pass from main())
	 * @return True if succeeded
	 */
	bool init(RobotParams_Type type, int argc, char *argv[]);

	/**
	 * @brief Commences the various control loops and tasks.
	 * @details This should be the *last* function the user calls at the end of main. It never returns.
	 * @return Never returns
	 */
	int begin();

	/**
	 * @brief Request a certain behavior change.
	 * To check if the request succeeded, please call checkBehavior()
	 * 
	 * @param field Subtype of request
	 * @param value Value corresponding to field
	 */
	void requestBehavior(BehavField_t field, uint32_t value);

	/**
	 * @brief Check the current behavior state (is decoded from modes)
	 * 
	 * @param field Behavior subtype to query
	 * @return uint32_t Value corresponding to field
	 */
	inline uint32_t checkBehavior(BehavField_t field) const {
		return decodeBehavior(&mode_, field);
	}

	/**
	 * @brief This function is public so it can be called from sdkUpdateA()
	 */
	void updatePeripherals() {
		for (auto p : peripherals)
			p->update(this);
	}

	/**
	 * @brief Get the limb position
	 * 
	 * @param i Limb index
	 * @return VecM_t Cartesian end-effector coordinates x,y,z in a frame centered at the hip and parallel to the body.
	 */
	inline VecM_t getLimbPosition(int i) const {
		return MCVecM(S.limbs[i].position);
	}

	/**
	 * @brief Get the limb velocity
	 * 
	 * @param i Leg index
	 * @return VecM_t Cartesian end-effector coordinates x,y,z in a frame centered at the hip and parallel to the body.
	 */
	inline VecM_t getLimbVelocity(int i) const {
		return MCVecM(S.limbs[i].velocity);
	}

	/**
	 * @brief Get the Jacobian of the leg forward kinematics.
	 * 
	 * @param i Leg index
	 * @return MatMN_t Matrix with row major internal layout. E.g. element (0,2) = dx/dq2, where q is indexed by HHIPJ, KNEEJ, ABDUCTIONJ.
	 */
	inline MatMN_t getLimbJac(int i) const {
		return MCMatMN(S.limbs[i].Jac);
	}

	inline float getJointPosition(int legi, int jointInLegi) const {
		return S.limbs[legi].q[jointInLegi];
	}

	/**
	 * @brief Get the external torque estimate (output of momentum observer)
	 * 
	 * @param i Leg index
	 * @return VecN_t Vector of torque estimate (N-m) on each joint. The index into this vector is HIPJ, KNEEJ, ABDUCTIONJ.
	 */
	inline VecN_t getJointExternalTorqueEstimate(int i) const {
		return MCVecN(S.limbs[i].torqueExt);
	}

	/**
	 * @brief Command individual joints in torque or current mode. Each leg must either be commanded in the end-effector space using setLimbForce() etc., or in the joint space using this command or setJointPositions().
	 * 
	 * @param legi Leg index
	 * @param torqueMode True to command in N-m, false to command current in A
	 * @param u 
	 */
	inline void setJointOpenLoops(int legi, bool torqueMode, const VecN_t &u) {
		C.limbs[legi].mode = torqueMode ? LimbCmdMode_JOINT_TORQUE : LimbCmdMode_JOINT_CURRENT;
		MVecN(C.limbs[legi].feedforward) = u;
	}

	/**
	 * @brief Command individual joint positions in a leg. Each leg must either be commanded in the end-effector space using setLimbForce() etc., or in the joint space using this command or setJointOpenLoops().
	 * 
	 * @param legi Leg index
	 * @param torqueMode True if the gain is in N-m/rad etc., false if the gain is in A/rad etc.
	 * @param positionSetpoints Joint position setpoints in the end-effector coordinates
	 * @param kp Proportional gain in N-m/rad etc.
	 * @param kd Dissipative gain
	 */
	inline void setJointPositions(int legi, bool torqueMode, const VecN_t &positionSetpoints, const VecN_t &kp, const VecM_t &kd) {
		C.limbs[legi].mode = torqueMode ? LimbCmdMode_JOINT_POSITION_OVER_TORQUE : LimbCmdMode_JOINT_POSITION_OVER_CURRENT;
		MVecN(C.limbs[legi].feedforward) = positionSetpoints; // this value is used to hold the setpoint
		MVecN(C.limbs[legi].kp) = kp;
		MVecN(C.limbs[legi].kd) = kd;
	}

	/**
	 * @brief Set a limb force command (assumes joints in torque mode). `u = ff - kp * position - kd * velocity`
	 * 
	 * @param ff Feedforward torque (can include `kp * position_desired + kd * velocity_desired`)
	 * @param kp Position tracking gain
	 * @param kd Velocity tracking gain
	 */
	void setLimbForce(int i, const VecM_t &ff, const VecM_t &kp = VecM_t::Zero(), const VecM_t &kd = VecM_t::Zero());
	/**
	 * @brief Set a limb position; `u = kp * (setpoint - position) - kd * velocity`
	 */
	inline void setLimbPosition(int i, const VecM_t &setpoint, const VecM_t &kp, const VecM_t &kd) {
		setLimbForce(i, kp.cwiseProduct(setpoint), kp, kd);
	}
	inline void setLimbPosition(int i, const VecM_t &setpoint, float kp, float kd) {
		setLimbPosition(i, setpoint, VecM_t::Constant(kp), VecM_t::Constant(kd));
	}
	/**
	 * @brief Set a limb position; `u = kp * (setpoint - position) - kd * velocity`
	 */
	inline void setLimbForce(int i, const VecM_t &setpoint, float kp, float kd) {
		setLimbForce(i, kp * setpoint, VecM_t::Constant(kp), VecM_t::Constant(kd));
	}

	inline void setLimbForce(int i, const float *ff, const float *kp, const float *kd) {
		setLimbForce(i, MCVecM(ff), MCVecM(kp), MCVecM(kd));
	}

	/**
	 * @brief Call after setting standard limb commands to set advanced options. 
	 * mode8[0] = LSB = standard robot commands. Higher bytes described in arguments
	 * 
	 * @param i Limb number
	 * @param ji Joint index within limb
	 * @param value Bypass value
	 * @param zeroCurrent Set true to override the limb command and set zero current
	 * @param bypassMode Send direct commands to motor controller for diagnostics (bypasses other commands completely)
	 */
	void setJointAdvanced(int i, int ji, float value = 0, bool bypassCurrent = false, uint8_t bypassMode = 0);

	/**
	 * @brief Runs at CONTROL_RATE checking for all pending behavior transitions 
	 */
	void behaviorUpdate();

	// not inheriting from this to keep it isolated
	RobotParams P = RobotParams_init_zero;
	// These have to be public for sdkUpdateA to modify them
	RobotState S = RobotState_init_zero;
	RobotCmd C = RobotCmd_init_zero;

	HardwareConfig hardwareConfig = HardwareConfig_init_zero;
	BehaviorConfig behaviorConfig = BehaviorConfig_init_zero;
	// Use push_back etc.
	std::vector<Behavior *> behaviors;
	std::vector<Peripheral *> peripherals;
	
	void estopUpdate();

	/**
	 * @brief Call to disable all the robot joints (like in an estop event). It is better to request an e-stop.
	 */
	void disableAllJoints();

	// Corresponding to SYS_STATUS bitmaps
	/**
	 * @brief It represents the state of an item if it is present, enabled and operating correctly.
	 * See https://mavlink.io/en/messages/common.html#MAV_SYS_STATUS_SENSOR
	 * Each of these represent a bit in the elements of this enum
	 */
	struct DiagnosticBitmaps_t {
		uint32_t present, enabled, health;
	};
	
	/**
	 * @brief Returns the system diagnostic information. See DiagnosticBitmaps_t for more information.
	 * Usage examples: 
	 * 1. imuGoodBit = ((R->diagnostics().health & MAV_SYS_STATUS_AHRS) != 0);
	 * 
	 * 2.	auto diag = robot.diagnostics();
  	 *		printf("%x %x\t", ~diag.health & diag.present, robot.P.version);
	 * 
	 * @return DiagnosticBitmaps_t 
	 */
	DiagnosticBitmaps_t diagnostics() const { return diag_; }
	// Call to report errors
	void reportHardwareHealth(uint32_t mask, bool good);
	void reportHardwareEnabled(uint32_t mask, bool good) {
		if (good)
			diag_.enabled |= mask;
		else
			diag_.enabled &= ~mask;
	}

	/**
	 * @brief Call to check if there is some hardware health issue
	 * 
	 * @return true Fine
	 * @return false Problem
	 */
	bool hardwareHealthGood();
	
	inline Mode_t mode() const { return mode_; }

	bool isSignalAllowed(BehavField_t field) const;
	std::atomic_bool keep_alive{true};

private:
	struct ModeRequest_t {
		BehavField_t field;
		uint32_t value;
		bool ongoing = false; // true when we are busy with this transition but it isn't complete
		ModeRequest_t(BehavField_t field, uint32_t value) : field(field), value(value), ongoing(false) {}
	};
	std::queue<ModeRequest_t> reqs_;

	// Mark a request as completed and remove from the queue
	inline void completeRequest() {
		if (!reqs_.empty()) {
			auto req = reqs_.front();
			forceBehavior(req.field, req.value);
			reqs_.pop();
		}
	}

	// Make sure incompatible modes are set appropriately
	inline void ensure(BehavField_t field, uint32_t value) {
#if defined(ARCH_obc)
		// printf("[ctrl_code][BehavSupervisor-ensure] field => %d \t valdes => %d \t val => %d \n", field, value, checkBehavior(field));
#endif
		if (checkBehavior(field) != value)
			reqs_.emplace(field, value);
	}

	uint32_t firstUpdate = 0;

	// Check sensors and set status errors
	void sensorDataCheck();

	// Initiate recovery if needed
	void initiateRecovery();

	void firstTimeBehavior();
	void addDefaultBehaviors();
	void initMotorControllers(uint32_t tstart, uint32_t duration);

	bool jointTempCheck() const;
	bool jointNegativePowerCheck() const;

	void recoveryBehaviorUpdate();
	void actionBehaviorUpdate();
	void pendingModesUpdate(); // removes stuff from the queue when complete

	// Forces update to the state
	void forceBehavior(BehavField_t field, uint32_t value) {
		encodeBehavior(&mode_, field, value);
	}
	
	// Called to request resetting modes
	void ensureSignalsReset(bool blindStairs);

  // Robot echoes back what behavior is selected
  Mode_t mode_;
  // Used when behavior transitions in or out recovery mode
  uint32_t prevBehaviorAction_ = 0;
	// For diagnostics; corresponding to SYS_STATUS
	DiagnosticBitmaps_t diag_;
};

/** @} */ // end of addtogroup

}
