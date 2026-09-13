import argparse

from sr_agent._vendor.llmsr_bench.algorithms.my_sr_agent import update_parser


def test_default_benchmark_tools_exclude_self_modification():
    parser = update_parser(argparse.ArgumentParser())
    tools = parser.get_default("tools")
    assert "edit_tool" not in tools
    assert "harmonic_interaction_fit" not in tools
    assert parser.parse_args(["--tools", "harmonic_interaction_fit"]).tools == [
        "harmonic_interaction_fit"
    ]
