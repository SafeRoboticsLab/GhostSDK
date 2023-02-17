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
#include <mblink/mblink_socket.hpp>
#include <iostream>
#include <cstring>

namespace gr {

SOCKET createSocket(std::string address, int port, bool bcast, bool input, struct sockaddr_in *tx_addr)
{
	struct timeval tv;
#ifdef _WIN32
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
	// Create the socket
	SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);

	// for socket options
	constexpr int enable = 1;

	if (input)
	{
		if (setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&enable, sizeof(int)) == SOCKET_ERROR)
		{
			std::cerr << "MBLink::createSocket setsockopt(SO_REUSEADDR) failed\n";
			return -1;
		}
#ifndef _WIN32
		if (setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char *)&enable, sizeof(int)) == SOCKET_ERROR) {
			std::cerr << "MBLink::createSocket setsockopt(SO_REUSEPORT) failed\n";
			return -1;
		}
#endif
		// Bind to the MB port
		struct sockaddr_in inAddr;
		inAddr.sin_family = AF_INET;
		inAddr.sin_port = htons(port);
		inAddr.sin_addr.s_addr = INADDR_ANY;
		if (bind(s, (sockaddr *)&inAddr, sizeof(inAddr)) == SOCKET_ERROR)
		{
			std::cerr << "MBLink::createSocket bind failed\n";
			return -1;
		}

		// Timeout to detect if the receive thread will hang
		tv.tv_sec = 1;
		tv.tv_usec = 0;
#ifndef _WIN32
		if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv)) < 0)
		{
			std::cerr << "MBLink::createSocket set receiver timeout failed\n";
			return -1;
		}
#endif
	}
	else
	{
		if (bcast)
		{
			// Ask OS for broadcast permission
			if (setsockopt(s, SOL_SOCKET, SO_BROADCAST, (const char *)&enable, sizeof(int)) == SOCKET_ERROR)
			{
				std::cerr << "MBLink::createSocket setsockopt(SO_BROADCAST) failed\n";
				return -1;
			}
		}

		// Set destination address to reuse
		std::memset(tx_addr, 0, sizeof(struct sockaddr_in));
		tx_addr->sin_family = AF_INET;
		tx_addr->sin_port = htons(port);
		tx_addr->sin_addr.s_addr = inet_addr(address.c_str());
	}
	return s;
}

} // namespace gr
