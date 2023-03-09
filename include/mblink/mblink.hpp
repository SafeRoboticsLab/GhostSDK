/**
MIT License (modified)

Copyright (c) 2021 Ghost Robotics
Authors:
Avik De <avik@ghostrobotics.io>
Tom Jacobs <tom.jacobs@ghostrobotics.io>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this **file** (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

// Protocol
#include "mblink_protocol.hpp"
#include "mblink_socket.hpp"
#include <thread>

namespace gr
{

/** @addtogroup mblink mblink
 * @brief C++/Python library to communicate with the robot via mavlink
 *  @{
 */

/**
 * @brief Class adding UDP socket communications to instantiate the mblink protocol
 */
class MBLink : public MBLinkProtocol
{
public:
	MBLink() : rxKeepRunning(true) {}

	~MBLink()
	{
		if (rxThread.joinable())
		{
			rxThread.join();
		}
	}

	/**
	 * @brief Initialize from command line arguments
	 * 
	 * @param argc 
	 * @param argv 
	 * @return int 
	 */
	int start(int argc, char **argv);

	/**
	 * @brief Start MAVLink connection socket and check IP address
	 * 
	 * @param address defaults to broadcast to robot subnet
	 * @param verbose Print messages
	 * @param rx_port 0=>default
	 */
	int start(bool sim, bool verbose, uint16_t rx_port);

	/**
	 * @brief This function must be called to start the reception thread. 
	 * This should be called after start()
	 * Get/set params will not work till after this is called.
	 */
	void rxstart();

	/**
	 * @brief Stop the RX thread. Usually this should be called when the program is exiting.
	 * 
	 * @param waitForExit If true, waits for the RX thread to join.
	 * @return RxAccumData_t Return all the accumulated data that has been collected, for logging.
	 */
	RxAccumData_t rxstop(bool waitForExit = true);

	/**
	 * @brief Read a parameter. This function will block till a new value is read, but not hog CPU.
	 * 
	 * @param name param name
	 * @return float param value
	 */
	float readParam(const std::string &name, bool print_on_failure);

	/**
	 * @brief Set a named parameter on the mainboard with retries for robustness. This function will block till done.
	 * 
	 * @param name param name
	 * @param val new value
	 */
	void setRetry(const std::string &name, float val);

	/**
	 * @brief Set a planner parameter. This function will block till a new value is read to confirm, but not hog CPU.
	 * 
	 * @param name string parameter name
	 * @param val value
	 * @return Eigen::VectorXf The parameter value as read from the mainboard
	 */
	Eigen::VectorXf setPlannerParam(const std::string &name, const Eigen::VectorXf &val) {
		return readWritePlannerParam(name, val, true);
	}

	/**
	 * @brief Read a planner parameter. This function will block till a new value is read, but not hog CPU.
	 * 
	 * @param name string parameter name
	 * @return Eigen::VectorXf The parameter value as read from the mainboard
	 */
	Eigen::VectorXf readPlannerParam(const std::string &name) {
		return readWritePlannerParam(name, {}, false);
	}

	/**
	 * @brief Set a mode and confirm if it worked or times out
	 * 
	 * @param fieldName Field name
	 * @param val new value
	 * @param timeoutMS timeout in ms
	 * @return true It worked
	 * @return false Timed out
	 */
	bool ensureMode(const std::string &fieldName, uint32_t val, uint32_t timeoutMS);

	/**
	 * @brief Send a joint self check command. WARNING: do not send without Ghost Robotics advisement.
	 * 
	 * @param action See low-level MAVLink documentation page
	 * @param param See low-level MAVLink documentation page
	 */
	void selfCheck(int16_t action, int16_t param);

	/**
	 * @brief Get all the rxdata. It will make a copy if there is new data.
	 * 
	 * @return RxData_t 
	 */
	RxData_t get();

	/// avg rate at which new data is being updated
	float avgRate = 0;

protected:
	// Addresses
	std::string LOCAL_ADDR = "127.0.0.1";
	std::string MB_ADDR = "192.168.168.5";
	std::string BCAST_ADDR = "192.168.168.255";
	const uint16_t TX_PORT = 14999;
	const uint16_t RX_PORT = 15000;
	const uint16_t APP_TX_PORT = 16000;
	const uint16_t APP_RX_PORT = 15999;

	bool verbose = true;

	// Message buffer
	constexpr static int BUFFER_LENGTH = 1024;
	unsigned char buffer[BUFFER_LENGTH];
	struct sockaddr_in mbAddr;
	SOCKET txSocket, rxSocket;
	struct timeval tv;

	bool ipCheck();

	/**
	 * @brief Basic TX implementation which creates a UDP packet from each message
	 */
	virtual int queueMessage(uint16_t messageLength);

	void rxThreadTask();

	std::atomic_bool rxKeepRunning;
	std::thread rxThread;
	
	Eigen::VectorXf readWritePlannerParam(const std::string &name, const Eigen::VectorXf &val, bool write);
};

/** @} */ // end of addtogroup

} // namespace gr
