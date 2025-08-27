import cv2
import torch
import torchvision.transforms as tf

from data_utils import image_transform
from model import RobotResNetRegressor
from real_camera_utils_new import Camera

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from scipy.spatial.transform import Rotation as R

import rospy

def load_model(checkpoint_path, device="cuda"):
    model = RobotResNetRegressor(
        resnet_type="resnet_18", output_dim=8, input_channels=3, dropout_rate=0.05
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    camera = Camera(camera_type="3rd")
    model = load_model("checkpoint/model_epoch2.pth", device="cuda")
    frequency = 20
    dt = 1 / 20
    T = 30
    image_transform = tf.Compose([tf.Resize((224, 224)), tf.ToTensor()])

    image = camera.capture()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_transform(image).to(device)
    
    output = model(image)
    
    pose = RigidTransform(
        translation = output[:3],
        rotation = RigidTransform.rotation_from_quaternion(output[3:8]),
        from_frame = 'franka_tool',
        to_frame = 'world'
    )
    
    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)
    
    rospy.loginfo('Publishing pose trajectory...')
    
    fa.goto_pose(pose, duration = T, dynamic=True, buffer_time=30)

    init_time = rospy.Time.now().to_time()
    previous_gripper_state = None
    index = 1
    
    while True:
        index +=1
        timestamp = rospy.Time.now().to_time() - init_time
        
        image = camera.capture()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transform(image).to(device)
        
        output = model(image)
        
        pose = RigidTransform(
            translation = output[:3],
            rotation = RigidTransform.rotation_from_quaternion(output[3:8]),
            from_frame = 'franka_tool',
            to_frame = 'world'
        )
        
        current_gripper_state = output[7]
        
        # 发布位置指令
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=index, timestamp=timestamp,
            position=pose.translation, 
            quaternion=pose.quaternion
        )
        
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
        )
        
        pub.publish(ros_msg)
        rospy.loginfo('Publishing: ID {} - Gripper: {}'.format(index, 'CLOSED' if current_gripper_state else 'OPEN'))
        
        # 检查是否需要改变夹爪状态
        if previous_gripper_state is None or current_gripper_state != previous_gripper_state:
            rospy.loginfo('Changing gripper state to: {}'.format('CLOSED' if current_gripper_state else 'OPEN'))
            
            if current_gripper_state:  # True = 夹爪关闭
                fa.close_gripper()
            else:  # False = 夹爪打开
                fa.open_gripper()
            
            # 等待夹爪动作完成（重要！）
            rospy.sleep(1.0)  # 给夹爪足够时间完成动作
        previous_gripper_state = current_gripper_state
        rate.sleep()
        
    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')