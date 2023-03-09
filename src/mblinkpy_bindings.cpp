/*
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Confidential, proprietary and/or trade secret materials.
 * No Distribution without prior written approval.
 *
 * Written by Avik De <avik@ghostrobotics.io>
 *
 * Handles the protocol between the mainboard and any higher-level component
 * 
 * Updated: April 2020
 *
 */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
namespace py = pybind11;

#include <mblink/mblink.hpp>
#include <mblink/app_interface.hpp>
#include <modemgr_client.hpp>

using namespace gr;

PYBIND11_MODULE(grmblinkpy, m)
{
	m.doc() = "Handles the protocol between the mainboard and any higher-level component";

	py::class_<MBLink> mblink(m, "MBLink");

	mblink.def(py::init<>(), "Defaults")
		.def("start", py::overload_cast<bool, bool, uint16_t>(&MBLink::start), "Start")
		.def("get", &MBLink::get, "Dict of received data (old infrastructure legacy)")
		.def("ensureMode", &MBLink::ensureMode, "")
		.def("selfCheck", &MBLink::selfCheck, "")
		.def("requestMode", &MBLink::requestMode, "")
		.def("getMode", &MBLink::getMode, "")
		.def("sendControl", &MBLink::sendControl, "")
		.def("sendPose", &MBLink::sendPose, "")
		.def("setParam", &MBLink::setParam, "")
		.def("setPlannerParam", &MBLink::setPlannerParam, "")
		.def("readPlannerParam", &MBLink::readPlannerParam, "")
		.def("setRetry", &MBLink::setRetry, "")
		.def("readParam", &MBLink::readParam, "")
		.def("sendSE2Twist", &MBLink::sendSE2Twist, "")
		.def("sendToSim", &MBLink::sendToSim, "")
		.def("sendUser", &MBLink::sendUser, "")
		.def("sendGoal", &MBLink::sendGoal, "")
		.def("rxstart", &MBLink::rxstart, "")
		.def_readonly("avgRate", &MBLink::avgRate, "")
    .def_property("stance_leg_03", &MBLink::isStanceLeg03, nullptr)
		.def("rxstop", &MBLink::rxstop, "", py::arg("waitForExit") = true);
	
	py::class_<ModeMgrClient> mmgr(m, "ModeMgrClient");
	mmgr.def(py::init<>(), "Defaults")
		.def("getAll", &ModeMgrClient::getAll, "")
		.def("received", &ModeMgrClient::received, "")
		.def("set", &ModeMgrClient::set, "")
		.def("selfCheck", &ModeMgrClient::selfCheck, "");
	
  py::class_<AppInterface>(m, "AppInterface")
		.def(py::init<>(), "Defaults")
		.def("rxstart", &AppInterface::rxstart, "")
		.def("rxstop", &AppInterface::rxstop, "")
		.def("send", &AppInterface::send, "");

  py::enum_<MAV_FRAME>(m, "Frame")
    .value("GLOBAL", MAV_FRAME_GLOBAL)
    .value("BODY_FLU", MAV_FRAME_BODY_FLU)
    .export_values();
}
