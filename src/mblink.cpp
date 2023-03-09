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
#include <mblink/mblink.hpp>
// Includes
#include <set>
#include <string>
// Network
#ifdef _MSC_VER

#else
#include <ifaddrs.h>
#include <fstream> // to check if WSL
#endif
#include <iomanip>
#include <iostream>

// Using
using std::string;
using std::set;
using std::chrono::steady_clock;
using std::chrono::duration_cast;

namespace gr
{

int MBLink::queueMessage(uint16_t messageLength)
{
	int payload_length = mavlink_msg_to_send_buffer(buffer, &msg);

	// Send
	int send_length = sendto(txSocket, (char *)buffer, payload_length, 0, (struct sockaddr *)&mbAddr, sizeof(mbAddr));
	if (send_length <= 0)
		printf("**** Error sending message to robot.\n");

	return (send_length == payload_length);
}

set<string> getIPs(bool print = false);

static bool isWSL()
{
#ifdef __linux__
	// both Linux and WSL go here
	// https://stackoverflow.com/questions/38086185/how-to-check-if-a-program-is-run-in-bash-on-ubuntu-on-windows-and-not-just-plain
	std::ifstream osrelease("/proc/sys/kernel/osrelease");
	// read the whole file into a string
	std::string str((std::istreambuf_iterator<char>(osrelease)), std::istreambuf_iterator<char>());
	if (str.find("icrosoft") != std::string::npos)
	{
		return true;
	}
#else
	return false;
#endif
}

bool MBLink::ipCheck()
{
#ifndef _WIN32
	// Check IP address. You won't send any UDP packets if you don't have an IP address on the same subnet.
	bool okIP = false;
	set<string> IPs = getIPs();
	set<string>::iterator it = IPs.begin();
	while (it != IPs.end())
	{
		// Found a correct IP address?
		string IP = *it;
		if (IP.find("192.168.168.") != string::npos)
			okIP = true;
		it++;
	}
	if (!okIP)
	{
		printf("\n");
		printf("**** Error: No valid IP address, so we can't send packets to robot.\n");
		printf("Check your computer's network adaptor (WiFi or ethernet) is connected to the robot.\n");
		printf("Configure the network interface connected to the robot to be DHCP, or static IP: 192.168.168.100.\n");
		printf("Current IP addresses:\n");
		getIPs(true);
		printf("\n");
		return false;
	}
#endif
	return true;
}

int MBLink::start(bool sim, bool verbose, uint16_t rx_port_alt)
{
	// if (!(sim || ipCheck()))
	// 	return -1;

	bool isWin32 = false;
#ifdef _WIN32
	isWin32 = true;
#endif
#ifdef __linux__
	isWin32 = isWSL();
#endif
	if (!isWin32)
	{
		MB_ADDR = BCAST_ADDR;
	}
	std::string rxAddr = sim ? LOCAL_ADDR : MB_ADDR;
	std::string txAddr = sim ? LOCAL_ADDR : BCAST_ADDR;

	// Create TX and RX sockets
	txSocket = createSocket(txAddr.c_str(), TX_PORT, true, false, &mbAddr);
	rxSocket = createSocket(rxAddr.c_str(), rx_port_alt>0 ? rx_port_alt : RX_PORT, false, true, NULL);

	this->verbose = verbose;

	// Success
	return 0;
}

int MBLink::start(int argc, char *argv[])
{
	if (argc > 1)
	{
		std::string arg1 = std::string(argv[1]);
		if (arg1 == "-s")
		{
			return start(true, true, 0);
		}
	}
	return start(false, true, 0);
}


void MBLink::rxstart()
{
	std::cout << std::setprecision(2) << std::fixed;
	rxThread = std::thread(&MBLink::rxThreadTask, this);
}

RxAccumData_t MBLink::rxstop(bool waitForExit)
{
	rxKeepRunning.exchange(false);
	if (waitForExit)
		rxThread.join();
	std::cout << "MBLink rxstop completed.\n";
	return rx_accum_;
}

void MBLink::rxThreadTask()
{
	char buf[10000];
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	int addrlen = sizeof(addr);

	float t0 = 0, avgdt = 0;
	steady_clock::time_point lastPrint = steady_clock::now();
	// Timeout to detect if the receive thread will hang
	tv.tv_sec = 1;
	tv.tv_usec = 0;

#ifdef _WIN32
	fd_set fds;
	int n;
	FD_ZERO(&fds);
	FD_SET(rxSocket, &fds);
#endif

	while (rxKeepRunning)
	{
#ifdef _WIN32
		// Detect mb disconnection (SO_RCVTIMEO replacement for win32)
		n = select((int)rxSocket, &fds, NULL, NULL, &tv);
		if (n <= 0)
		{
			std::cerr << "MBLink::rxThreadTask recvfrom error (most likely timeout).\n";
			rxKeepRunning.exchange(false);
			continue;
		}
#endif
		// perform the blocking recvfrom syscall
		int recvlen = recvfrom(rxSocket, buf, sizeof(buf), 0, (struct sockaddr *)&addr, (socklen_t *)&addrlen);

		if (recvlen <= 0)
		{
			std::cerr << "[MBLink::rxThreadTask] recvfrom error (most likely timeout)." << std::endl;
			rxKeepRunning.exchange(false);
			continue;
		}
		
		constexpr uint32_t start_word = 0x87654321;
		int numparsed = 0;
		if (memcmp(&buf[0], &start_word, 4) == 0) {
			newPlannerUDPMsg(&buf[4], recvlen-4);
		} else {
			conditionMutex.lock();
			numparsed = parse(buf, recvlen); // ignore high-rate messages
			conditionMutex.unlock();
		}

		float t1 = rxdata["y"][rxdata["y"].size() - 1];
		if (numparsed > 0 && t > 1e-3f) {
			avgdt += 0.1f * (t1 - t0 - avgdt);
			this->avgRate = 1 / avgdt;
			t0 = t1;
		}

		if (verbose)
		{
			auto now = steady_clock::now();
			bool pl_running = t - last_pl_time_ < 0.5f && ctrldata_["pl_y"].size() > 0;
			int print_interval = pl_running ? 500 : 1000;
			if (duration_cast<std::chrono::milliseconds>(now - lastPrint).count() > print_interval)
			{
				std::cout << "[mblink rx]\tt=" << t0 << "\trate=" << 1 / avgdt;
				if (pl_running) {
					std::cout << "\tq=" << ctrldata_["pl_y"].head<6>().transpose() << "\tu=" << ctrldata_["u"].head<3>().transpose() << "\tgoal=" << ctrldata_["goal"].transpose() << "\tplanner_cmd=" << mmgr.get(Behav_PLANNER_CMD);
				} else {
					// only have mb data
					std::cout << "\tstatus=" << rxdata["behavior"][2] << "\tvoltage(batt,mot)=" << rxdata["voltage"].transpose();
				}
				std::cout << std::endl;
				lastPrint = now;
			}
		}
	}
	std::unique_lock<std::mutex> lk(conditionMutex);
	cv.notify_one(); // so that we can quit
}

float MBLink::readParam(const std::string &name, bool print_on_failure)
{
	float retval = 0;
	std::chrono::milliseconds timeout(10);
	requestParam(name);
	std::unique_lock<std::mutex> lk(paramMutex);
	
	if (paramcv.wait_for(lk, timeout, [this] { return rxdata["param_value"].size() > 0; })) {
		retval = rxdata["param_value"][0];
	} else if (print_on_failure) {
		std::cerr << "[MBLink::readParam] did not receive param " << name << std::endl;
	}
	return retval;
}

void MBLink::setRetry(const std::string &name, float val)
{
	int nTries = 10;
	std::chrono::milliseconds paramwait(20);
	for (int i = 0; i < 10; ++i)
	{
		setParam(name, val);
		std::this_thread::sleep_for(paramwait);
		if (std::abs(readParam(name, false) - val) < 1e-4f)
		{
			std::cout << name << " = " << val << std::endl;
			return;
		}
	}
	std::cerr << "[MBLink::setRetry] error: " << name << " not set" << std::endl;
}

Eigen::VectorXf MBLink::readWritePlannerParam(const std::string &name, const Eigen::VectorXf &val, bool write) {
	Eigen::VectorXf retval;
	std::chrono::milliseconds timeout(20);
	// to read, clear the param value (can check its size for reception)
	paramMutex.lock();
	rxdata["param_value"].resize(0);
	paramMutex.unlock();

	// Send the mavlink message
	sendPlannerParam(name, val, write);
	// Now wait for response
	std::unique_lock<std::mutex> lk(paramMutex, std::defer_lock);
	lk.lock();

	if (paramcv.wait_for(lk, timeout, [this] { return rxdata["param_value"].size() > 0; })) {
		retval = rxdata["param_value"];
		if (retval.isApprox(val)) {
			std::cout << name << " = " << retval.transpose() << std::endl;
		} else if (write) {
			std::cerr << "[MBLink::readWritePlannerParam] Received " << name << " = " << retval.transpose() << " instead of " << val.transpose() << std::endl;
		}
	} else {
		std::cerr << "[MBLink::readWritePlannerParam] did not receive param " << name << std::endl;
	}
	return retval;
}

bool MBLink::ensureMode(const std::string &fieldName, uint32_t valdes, uint32_t timeoutMS)
{
	const uint32_t modewait = 50;
	std::chrono::milliseconds timeout(modewait);
	auto tosend = mmgr.set(fieldName, valdes);
	for (size_t i = 0; i < timeoutMS / modewait; ++i) {
		setModeRaw(std::get<0>(tosend), std::get<1>(tosend));
		std::unique_lock<std::mutex> lk(modeMutex);
		// Predicate that returns false when we should keep waiting
		if (modecv.wait_for(lk, timeout, [this, fieldName, valdes] { return mmgr.get(fieldName) == valdes; }))
			return true;
	}
	return false;
}

void MBLink::selfCheck(int16_t action, int16_t param) {
	auto tosend = mmgr.selfCheck(action, param);
	setModeRaw(std::get<0>(tosend), std::get<1>(tosend));
}

RxData_t MBLink::get()
{
	static RxData_t ret;
	std::unique_lock<std::mutex> lk(conditionMutex);
	cv.wait(lk, [this] { return newRxData.load() || !rxKeepRunning.load(); });
	// while (!newRxData && rxKeepRunning) { std::this_thread::sleep_for(std::chrono::microseconds(500)); }
	if (rxKeepRunning)
	{
		// means newRxData was true
		newRxData.exchange(false);
		// mutex.lock();
		ret = rxdata; // copy assignment
		// mutex.unlock();
	}
	return ret;
}

#ifndef _WIN32
// Return a list of IP addresses on system
set<string> getIPs(bool print)
{
	// Get all IP addresses on system
	set<string> IPs;
	struct ifaddrs *ifap, *ifa;
	struct sockaddr_in *sa;
	char *addr;
	getifaddrs (&ifap);
	for (ifa = ifap; ifa; ifa = ifa->ifa_next)
	{
		if (ifa->ifa_addr->sa_family==AF_INET)
		{
			sa = (struct sockaddr_in *) ifa->ifa_addr;
			addr = inet_ntoa(sa->sin_addr);
			IPs.insert(addr);
			if(print)
				printf("Address: %s\t\tInterface: %s\n", addr, ifa->ifa_name);
		}
	}
	freeifaddrs(ifap);
	return IPs;
}
#endif

} // namespace gr
