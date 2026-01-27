#!/usr/bin/env python3
"""
Simple flow diagram of LodeSTAR with Skip Connections
"""

def print_simple_flow():
    """Print a simple flow diagram"""
    
    print("=" * 60)
    print("LodeSTAR with Skip Connections - Flow Diagram")
    print("=" * 60)
    print()
    
    print("Input (1×H×W)")
    print("    │")
    print("    ▼")
    print("Block 0 (32×H×W) ──────────────────────────────────────────┐")
    print("    │                                                      │")
    print("    ▼                                                      │")
    print("Block 1 (32×H×W) ─────────────────────────────────────┐    │")
    print("    │                                                 │    │")
    print("    ▼                                                 │    │")
    print("Block 2 (64×H/2×W/2) [POOL] ──────────────────────┐   │    │")
    print("    │                                              │   │    │")
    print("    ▼                                              │   │    │")
    print("Block 3 (64×H/2×W/2) ──────────────────────────┐   │   │    │")
    print("    │                                          │   │   │    │")
    print("    ▼                                          │   │   │    │")
    print("Block 4 (64×H/2×W/2) ←────────────────────────┘   │   │    │")
    print("    │                                              │   │    │")
    print("    ▼                                              │   │    │")
    print("Block 5 (64×H/2×W/2) ←────────────────────────────┘   │    │")
    print("    │                                                  │    │")
    print("    ▼                                                  │    │")
    print("Block 6 (64×H/2×W/2) ←────────────────────────────────┘    │")
    print("    │                                                      │")
    print("    ▼                                                      │")
    print("Block 7 (64×H/2×W/2) ←────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("Block 8 (64×H/2×W/2)")
    print("    │")
    print("    ▼")
    print("Block 9 (3×H/2×W/2) [Δx, Δy, ρ]")
    print()
    
    print("Skip Connections:")
    print("• Block 0 → Block 4: Early spatial features")
    print("• Block 1 → Block 5: Pre-pooling features") 
    print("• Block 2 → Block 6: Across pooling bottleneck")
    print("• Block 3 → Block 7: Mid-level features")
    print("• Block 4 → Block 8: Processed features")
    print()
    
    print("Key Benefits:")
    print("• Preserves spatial resolution")
    print("• Improves gradient flow")
    print("• Combines multi-scale features")
    print("• Only 1.7% parameter increase")
    print("=" * 60)


if __name__ == "__main__":
    print_simple_flow()
