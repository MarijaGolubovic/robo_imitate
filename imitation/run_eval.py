import rclpy
from modelling.scd_agent import StateTimeredPickingAgent
from modelling.scd_env import PickScrewdriverSimEnv


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='PickScrewdriver')
    parser.add_argument('--num_restarts', type=int, default=0, help='Number of evaluation iterations')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)

    agent = StateTimeredPickingAgent()
    cmd_vel_publisher = PickScrewdriverSimEnv(agent=agent, num_restarts=parsed_args.num_restarts)
    try:
        rclpy.spin(cmd_vel_publisher)
    except RuntimeError as exc:
        pass
        #print(f"Success rate: {np.mean(exc.result)}")
        #print(exc.result)

    cmd_vel_publisher.get_logger().info(f"Finished publishing. Shutting down node...")
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
