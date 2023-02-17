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
#pragma once
#include "modemgr.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

namespace gr {

/** @addtogroup Modes Modes
 *  @{
 */

typedef std::vector<uint32_t> FieldVals_t;
typedef std::unordered_map<std::string, uint32_t> FieldMap_t;
typedef std::unordered_map<std::string, BehavField_t> FieldNameToEnum_t;
typedef std::tuple<uint8_t, uint32_t> RequestTuple_t;

/**
 * @brief Client code to interface with the mainboard modes. Used (for example) in the app.
 */
class ModeMgrClient {
public:
	// Construct
	ModeMgrClient();

	/**
	 * @brief This should be called when a packed mode (as bmode,cmode) is received from the mainboard via mblink.
	 */
	void received(uint8_t bmode, uint32_t cmode);

	inline int32_t get(BehavField_t field) const {
		return decodeBehavior(&mode_, field);
	}

	/**
	 * @brief After received() is called, this can be called to decode what was received and check the value of some field
	 * 
	 * @param fieldstr Field whose value we would like to check as a string
	 * @return int32_t -1 if field is not found, otherwise the value of the field
	 */
	int32_t get(const std::string &fieldstr) const;

	/**
	 * @brief Get all field values as a dictionary-type object
	 * 
	 * @return FieldMap_t 
	 */
	FieldMap_t getAll() const;

	// For now just synchronize the command to the robot
	inline void connect() {
		// TODO:
	}

	// Client -> robot

	RequestTuple_t set2(BehavField_t field, uint32_t val) const;

	/**
	 * @brief The client can set a mainboard mode using this function
	 * 
	 * @param fieldstr field as a string
	 * @param val value for the field
	 * @return RequestTuple_t This should be packed into the bmode,cmode fields and sent over mavlink. mblink does this.
	 */
	RequestTuple_t set(const std::string &fieldstr, uint32_t val) const;

	// Has to match receiving code in JoyMAVLink.hardwareRequest
	RequestTuple_t selfCheck(int16_t action, int16_t param) const;

	// Thread?
	// void setField(const std::string &field, uint32_t value, uint32_t timeout);

protected:
  Mode_t mode_;
	uint32_t val_ = 0; // dummy
	bool receivedOnce_ = false;

	FieldVals_t fieldVals_;
	FieldMap_t fieldMap_;
	FieldNameToEnum_t fieldFromStr_;
};

/** @} */ // end of addtogroup

} // namespace gr
