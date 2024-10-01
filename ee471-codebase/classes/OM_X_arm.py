# (c) 2024 S. Farzan, Electrical Engineering Department, Cal Poly
# OM_X_arm class for OpenManipulator-X Robot for EE 471
# This class/file should not need to be modified in any way for EE 471

import serial.tools.list_ports
from DX_XM430_W350 import DX_XM430_W350
from dynamixel_sdk import PortHandler, PacketHandler, GroupBulkWrite, GroupBulkRead, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD

"""
OM_X_arm class for the OpenManipulator-X Robot.
Abstracts the serial connection and read/write methods from the Robot class.
"""
class OM_X_arm:
    """
    Initialize the OM_X_arm class.
    Sets up the serial connection, motor IDs, and initializes the motors and gripper.
    """
    def __init__(self):
        self.motorsNum = 4
        self.motorIDs = [11, 12, 13, 14]
        self.gripperID = 15
        self.deviceName = self.find_device_name()

        print(f"Port #: {self.deviceName}")
        self.port_handler = PortHandler(self.deviceName)
        self.packet_handler = PacketHandler(DX_XM430_W350.PROTOCOL_VERSION)

        # Create array of motors
        self.motors = [DX_XM430_W350(self.port_handler, self.packet_handler, self.deviceName, motor_id) for motor_id in self.motorIDs]
        
        # Create Gripper and set operating mode/torque
        self.gripper = DX_XM430_W350(self.port_handler, self.packet_handler, self.deviceName, self.gripperID)
        
        if self.port_handler.openPort():
            print("Succeeded to open the port")
            # print(f"Port Ser: {self.port_handler.ser}")
        else:
            print("Failed to open the port")

        if self.port_handler.setBaudRate(DX_XM430_W350.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")

        self.groupBulkWrite = GroupBulkWrite(self.port_handler, self.packet_handler)
        self.groupBulkRead = GroupBulkRead(self.port_handler, self.packet_handler)

        self.gripper.set_operating_mode('position') # TODO
        self.gripper.toggle_torque(True) # TODO

        # Enable motors and set drive mode
        enable = 1
        disable = 0
        self.bulk_read_write(DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, [enable]*self.motorsNum)
        self.bulk_read_write(DX_XM430_W350.DRIVE_MODE_LEN, DX_XM430_W350.DRIVE_MODE, [DX_XM430_W350.TIME_PROF]*self.motorsNum)
        # dxl_mode_result = self.bulk_read_write(DX_XM430_W350.DRIVE_MODE_LEN, DX_XM430_W350.DRIVE_MODE)
        # if dxl_mode_result[0] != DX_XM430_W350.TIME_PROF:
        #     print("Failed to set the DRIVE_MODE to TIME_PROF")
        #     quit()
        self.bulk_read_write(DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, [disable]*self.motorsNum)

    """
    Finds the device name for the serial connection.
    Returns:
    str: The device name (e.g., 'COM3' for Windows, 'ttyUSB0' for Linux).
    Raises:
    Exception: If no serial devices are found.
    """
    def find_device_name(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if 'COM' in port.device and port.device != 'COM1':  # Skip COM1 # ttyUSB
                return port.device
        raise Exception("Failed to connect via serial, no devices found.")

    """
    Reads or writes messages of length n from/to the desired address for all joints.
    Parameters:
    n (int): The size in bytes of the message (1 for most settings, 2 for current, 4 for velocity/position).
    addr (int): Address of control table index to read from or write to.
    msgs (list of 1x4 int, optional): The messages (in bytes) to send to each joint, respectively. If not provided, a read operation is performed. If a single integer is provided, the same message will be sent to all four joints.
    Returns:
    list of int: The result of a bulk read, empty if bulk write.
    """
    def bulk_read_write(self, n, addr, msgs=None):
        if msgs is None:  # Bulk read
            results = []
            self.groupBulkRead.clearParam()

            for motor_id in self.motorIDs:
                dxl_addparam_result = self.groupBulkRead.addParam(motor_id, addr, n)

            self.groupBulkRead.txRxPacket()
            for motor_id in self.motorIDs:
                dxl_getdata_result = self.groupBulkRead.isAvailable(motor_id, addr, n)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupBulkRead getdata failed" % motor_id)
                    quit()
                result = self.groupBulkRead.getData(motor_id, addr, n)
                results.append(result)

            self.groupBulkRead.clearParam()
            return results
        else:  # Bulk write
            self.groupBulkWrite.clearParam()

            # Add value to the Bulk write parameter storage
            for i, motor_id in enumerate(self.motorIDs):
                # Allocate value into byte array
                if n == 4:
                    param_msg = [DXL_LOBYTE(DXL_LOWORD(msgs[i])),
                                 DXL_HIBYTE(DXL_LOWORD(msgs[i])),
                                 DXL_LOBYTE(DXL_HIWORD(msgs[i])),
                                 DXL_HIBYTE(DXL_HIWORD(msgs[i]))]
                elif n == 2:
                    param_msg = [DXL_LOBYTE(DXL_LOWORD(msgs[i])),
                                 DXL_HIBYTE(DXL_LOWORD(msgs[i]))]
                elif n == 1:
                    param_msg = [DXL_LOBYTE(DXL_LOWORD(msgs[i]))]
                else:
                    raise ValueError("Invalid number of bytes to write")

                dxl_addparam_result = self.groupBulkWrite.addParam(motor_id, addr, n, param_msg)
                if dxl_addparam_result != True:
                    print("[ID:%03d] groupBulkWrite addparam failed" % motor_id)
                    quit()

            # Bulk write
            dxl_comm_result = self.groupBulkWrite.txPacket()
            if dxl_comm_result != DX_XM430_W350.COMM_SUCCESS:
                print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))

            # Clear bulkwrite parameter storage
            self.groupBulkWrite.clearParam()
