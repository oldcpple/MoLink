"""
Utility functions for MoLink cross-node pipeline parallelism.
"""

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)


def extract_ip() -> str:
    """Extract the local IP address.

    Returns:
        The local IP address as a string.
    """
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't actually establish a connection
        st.connect(("10.255.255.255", 1))
        ip = st.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        st.close()

    return ip


def find_free_port(start_port: int = 50051, protocol: str = "tcp") -> int:
    """Find an available port for TCP/UDP on all interfaces.

    Args:
        start_port: The port number to start searching from.
        protocol: The protocol type ('tcp' or 'udp').

    Returns:
        An available port number.
    """
    ip = "0.0.0.0"
    port = start_port

    while True:
        try:
            if protocol == "tcp":
                sock_type = socket.SOCK_STREAM
            elif protocol == "udp":
                sock_type = socket.SOCK_DGRAM
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")

            with socket.socket(socket.AF_INET, sock_type) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((ip, port))
            return port

        except OSError:
            port += 1
            if port > 65535:
                raise RuntimeError("No available port found")


def get_grpc_options(max_message_size_mb: int = 200) -> List[Tuple[str, int]]:
    """Get gRPC channel/server options.

    Args:
        max_message_size_mb: Maximum message size in MB.

    Returns:
        List of gRPC options tuples.
    """
    max_size = max_message_size_mb * 1024 * 1024
    return [
        ("grpc.max_send_message_length", max_size),
        ("grpc.max_receive_message_length", max_size),
    ]


@dataclass
class NodeInfo:
    """Information about a node in the pipeline."""

    ip: str  # host:port
    start_layer: int
    end_layer: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ip": self.ip,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            ip=data["ip"],
            start_layer=data["start_layer"],
            end_layer=data["end_layer"],
        )


@dataclass
class PipelineTopology:
    """Manages the topology of the distributed pipeline.

    Tracks all nodes in the pipeline and their layer assignments.
    The nodes are ordered by their start_layer to ensure correct
    execution order in the pipeline.
    """

    head_ip: str
    start_layer: int
    end_layer: int
    node_pool: List[Dict[str, Any]] = field(default_factory=list)
    node_info_dict: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the topology with the head node."""
        # Add self as head node
        self.add_node(self.head_ip, self.start_layer, self.end_layer)

    def add_node(self, ip: str, start_layer: int, end_layer: int) -> None:
        """Add a node to the topology.

        Args:
            ip: The node's address (host:port).
            start_layer: First layer the node handles.
            end_layer: Last layer the node handles.
        """
        # Check if node already exists
        if ip in self.node_info_dict:
            logger.warning(f"Node {ip} already in topology, updating info")
            # Update existing node
            for node in self.node_pool:
                if node["ip"] == ip:
                    node["start_layer"] = start_layer
                    node["end_layer"] = end_layer
                    break
            self.node_info_dict[ip] = start_layer
            return

        self.node_pool.append(
            {
                "ip": ip,
                "start_layer": start_layer,
                "end_layer": end_layer,
            }
        )
        self.node_info_dict[ip] = start_layer

        logger.info(
            f"Node {ip} added to topology " f"(layers {start_layer}-{end_layer})"
        )

    def remove_node(self, ip: str) -> None:
        """Remove a node from the topology.

        Args:
            ip: The node's address to remove.
        """
        self.node_pool = [n for n in self.node_pool if n["ip"] != ip]
        self.node_info_dict.pop(ip, None)
        logger.info(f"Node {ip} removed from topology")

    def get_sorted_server_list(self) -> List[str]:
        """Get the list of servers sorted by their start layer.

        Returns:
            List of server addresses in pipeline order.
        """
        sorted_items = sorted(self.node_info_dict.items(), key=lambda item: item[1])
        return [ip for ip, _ in sorted_items]

    def get_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata for cross-node communication.

        Returns:
            Dictionary containing pipeline metadata.
        """
        server_list = self.get_sorted_server_list()
        return {
            "head": self.head_ip,
            "server_list": server_list,
        }

    def get_next_server(self, current_ip: str) -> Optional[str]:
        """Get the next server in the pipeline.

        Args:
            current_ip: The current node's address.

        Returns:
            The next server's address, or None if this is the last.
        """
        server_list = self.get_sorted_server_list()
        try:
            idx = server_list.index(current_ip)
            if idx < len(server_list) - 1:
                return server_list[idx + 1]
        except ValueError:
            logger.error(f"Node {current_ip} not found in server list")
        return None

    def is_last_server(self, current_ip: str) -> bool:
        """Check if the current server is the last in the pipeline.

        Args:
            current_ip: The current node's address.

        Returns:
            True if this is the last server.
        """
        server_list = self.get_sorted_server_list()
        return server_list and server_list[-1] == current_ip

    def get_pp_rank(self, ip: str) -> int:
        """Get the pipeline parallel rank for a given IP.

        Args:
            ip: The node's address.

        Returns:
            The pipeline parallel rank (0-indexed).
        """
        server_list = self.get_sorted_server_list()
        try:
            return server_list.index(ip)
        except ValueError:
            return 0

    def get_pp_size(self) -> int:
        """Get the total pipeline parallel size.

        Returns:
            Number of nodes in the pipeline.
        """
        return len(self.node_pool)


def serialize_metadata(metadata: Dict[str, Any]) -> bytes:
    """Serialize metadata to bytes for gRPC transmission.

    Args:
        metadata: The metadata dictionary.

    Returns:
        JSON-encoded bytes.
    """
    return json.dumps(metadata).encode("utf-8")


def deserialize_metadata(data: bytes) -> Dict[str, Any]:
    """Deserialize metadata from bytes.

    Args:
        data: JSON-encoded bytes.

    Returns:
        The metadata dictionary.
    """
    return json.loads(data.decode("utf-8"))
