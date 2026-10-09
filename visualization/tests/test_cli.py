"""The documented no-argument render command must select B and both A files."""
import unittest
from unittest.mock import patch

from visualization.__main__ import main


class CommandTests(unittest.TestCase):
    def test_render_defaults_and_deduplicates_figures(self):
        for arguments in (["figures", "render"], ["figures", "render", "B", "A", "A1"]):
            with self.subTest(arguments=arguments), patch("sys.argv", arguments), \
                    patch("visualization.__main__.load_config"), \
                    patch("visualization.__main__.runpy.run_path") as run:
                main()
                self.assertEqual([call.args[0].replace("\\", "/").rsplit("/", 1)[-1]
                                  for call in run.call_args_list],
                                 ["selection_distributions.py", "acoustic_active.py", "acoustic_ships.py"])


if __name__ == "__main__":
    unittest.main()
