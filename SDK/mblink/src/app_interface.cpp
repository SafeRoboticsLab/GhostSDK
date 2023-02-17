/**
MIT License (modified)

Copyright (c) 2021 Ghost Robotics
Authors:
Avik De <avik@ghostrobotics.io>

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
#include <mblink/app_interface.hpp>
#include <iostream>
#include <cstring>

namespace gr {

void AppInterface::rxThreadTask(const AppInterface::SetModeFunc_t &f_set_mode) {
	// Create RX socket
	rx_socket_ = createSocket("", APP_TX_PORT, false, true, NULL);
	// Create TX socket (but dest addr will be updated later)
	tx_socket_ = createSocket(app_IP_.c_str(), APP_RX_PORT, false, false, &tx_addr_);

	char buf[1024];
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	int addrlen = sizeof(addr);

	while (true) {
		// perform the blocking recvfrom syscall
		int recvlen = recvfrom(rx_socket_, buf, sizeof(buf), 0, (struct sockaddr *)&addr, (socklen_t *)&addrlen);

		if (recvlen <= 0 || (recvlen == 5 && memcmp(buf, CLOSE_STR, 5) == 0)) {
			std::cerr << "[AppInterface::rxThreadTask] stopping.\n";
			break;
		}
		
		// parse mavlink message
		for (size_t i = 0; i < recvlen; ++i) {
			if (mavlink_parse_char(rx_chan_, buf[i], &rx_msg_, &rx_status_) && rx_msg_.sysid==255 && rx_msg_.compid==190) {
				// Update the IP to where mavlink commands are coming from
				app_IP_ = inet_ntoa(addr.sin_addr);

				// std::cout << "[AppInterface::rxThreadTask] received sysid=" << (int)rx_msg_.sysid << ", compid=" << (int)rx_msg_.compid << ",len=" << (int)rx_msg_.len << "from IP=" << app_IP_ << std::endl;

				// Decode message and act on it
				switch (rx_msg_.msgid) {
				case MAVLINK_MSG_ID_SET_MODE:
					mavlink_msg_set_mode_decode(&rx_msg_, &set_mode_);
					f_set_mode(set_mode_.base_mode, set_mode_.custom_mode);
					break;
				
				default:
					break;
				}
			}
		}
	}
}

int AppInterface::send(float res, float xstart, const Eigen::VectorXf &zs, float mb_rate, float cam_rate) {
	constexpr size_t max_bytes = 58*4;
	static uint8_t bytes[max_bytes];

	// Encode heightmap as single byte per pixel
	if (zs.size() > max_bytes) {
		std::cerr << "[AppInterface::send] Vector size must be <" << max_bytes << std::endl;
		return -1;
	}

	// Encode: each byte is in -z_B in cm, typically between 0.14--2m.
	std::memset(bytes, 0, max_bytes);
	for (Eigen::Index i=0; i<zs.size(); ++i) {
		bytes[i] = std::isnan(zs[i]) || zs[i]>0 || zs[i] < -2.54 ? 255 : (uint8_t)std::round(-zs[i]*1e2);
	}

	// Encode header
	uint64_t time_usec = 0;
	MessageFromPlannerToAppHeader_t *phdr = (MessageFromPlannerToAppHeader_t *)&time_usec;
	phdr->res_mm = (uint8_t)(res*1e3);
	phdr->xstart_cm = (int16_t)(xstart*1e2);
	phdr->num_points = (uint16_t)zs.size();
	phdr->mb_rate = (uint8_t)mb_rate;
	phdr->cam_rate = (uint8_t)cam_rate;

	return sendMessage(mavlink_msg_debug_float_array_pack(system_id_, component_id_, &tx_msg_, time_usec, "", DEBUG_FLOAT_ARRAY_ID_FROM_PLANNER_TO_APP, (const float *)bytes));
}

int AppInterface::sendMessage(uint16_t msg_len) {
	if (tx_socket_ == 0) {
		return 0; // not initialized yet
	}

	static uint8_t buffer[1024];
	int payload_length = mavlink_msg_to_send_buffer(buffer, &tx_msg_);
	
	// Update the IP to where mavlink commands are coming from
	tx_addr_.sin_addr.s_addr = inet_addr(app_IP_.c_str());

	// Send
	int send_length = sendto(tx_socket_, (char *)buffer, payload_length, 0, (struct sockaddr *)&tx_addr_, sizeof(tx_addr_));
	if (send_length <= 0) {
		std::cerr << "[AppInterface::send] Error in sendto(). IP=" << app_IP_ << ", lengths=" << msg_len << "," << payload_length << std::endl;
	}

	return send_length;
}

void AppInterface::rxstop() {
	// send a message that closes the socket?
	SOCKET sclose = socket(AF_INET, SOCK_DGRAM, 0);
	sockaddr_in myself;
	myself.sin_family = AF_INET;
	myself.sin_port = htons(APP_TX_PORT);
	myself.sin_addr.s_addr = inet_addr("127.0.0.1");
	sendto(sclose, CLOSE_STR, 5, 0, (sockaddr *)&myself, sizeof(myself));
}

} // namespace gr
