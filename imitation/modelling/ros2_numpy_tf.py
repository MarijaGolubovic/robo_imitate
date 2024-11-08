import geometry_msgs.msg
import transforms3d as t3d
import numpy as np


def ros2numpy(pose):
    xyz = None
    q = None

    if isinstance(pose, geometry_msgs.msg.PoseStamped):
        pose = pose.pose
    elif isinstance(pose, geometry_msgs.msg.TransformStamped):
        pose = pose.transform

    if isinstance(pose, geometry_msgs.msg.Pose):
        xyz = [pose.position.x, pose.position.y, pose.position.z]
        q = [
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
    elif isinstance(pose, geometry_msgs.msg.Transform):
        xyz = [pose.translation.x, pose.translation.y, pose.translation.z]
        q = [pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z]
    else:
        raise ValueError("Unknown type")

    transform = t3d.affines.compose(xyz, t3d.quaternions.quat2mat(q), [1, 1, 1])
    return transform


def numpy2ros(transform, ros_type, frame_id="", stamp=None):
    xyz = transform[:3, 3]
    q = t3d.quaternions.mat2quat(transform[:3, :3])

    message = None
    if ros_type == geometry_msgs.msg.Pose:
        message = geometry_msgs.msg.Pose()
        message.position.x = xyz[0]
        message.position.y = xyz[1]
        message.position.z = xyz[2]
        message.orientation.w = q[0]
        message.orientation.x = q[1]
        message.orientation.y = q[2]
        message.orientation.z = q[3]
    elif ros_type == geometry_msgs.msg.Transform:
        message = geometry_msgs.msg.Transform()
        message.translation.x = xyz[0]
        message.translation.y = xyz[1]
        message.translation.z = xyz[2]
        message.rotation.w = q[0]
        message.rotation.x = q[1]
        message.rotation.y = q[2]
        message.rotation.z = q[3]
    elif ros_type == geometry_msgs.msg.PoseStamped:
        message = geometry_msgs.msg.PoseStamped()
        message.header.frame_id = frame_id
        message.header.stamp = stamp
        message.pose = numpy2ros(transform, geometry_msgs.msg.Pose)
    elif ros_type == geometry_msgs.msg.TransformStamped:
        message = geometry_msgs.msg.TransformStamped()
        message.header.frame_id = frame_id
        message.header.stamp = stamp
        message.transform = numpy2ros(transform, geometry_msgs.msg.Transform)
    else:
        raise ValueError("Unknown type")

    return message


def test():
    msg = geometry_msgs.msg.Pose()

    transform = ros2numpy(msg)
    assert np.allclose(transform, np.eye(4))

    transform = transform @ t3d.affines.compose(
        [0, 0, 0], t3d.euler.euler2mat(0, 0, np.pi / 2), [1, 1, 1]
    )
    msg = numpy2ros(transform, geometry_msgs.msg.Pose)

    assert np.isclose(msg.orientation.w, np.sqrt(2) / 2)
    assert np.isclose(msg.orientation.z, np.sqrt(2) / 2)


if __name__ == "__main__":
    test()
