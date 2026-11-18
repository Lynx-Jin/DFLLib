"""Verify parity when the local and distribution repositories are side by side."""

import ast
from pathlib import Path
import unittest


class SecurityAlignmentTest(unittest.TestCase):
    def test_shared_security_sources_and_training_hooks_match(self):
        repository = Path(__file__).resolve().parents[2]
        other_name = "DFLlib" if repository.name == "PFLlib-master" else "PFLlib-master"
        other = repository.parent / other_name
        if not other.is_dir():
            self.skipTest("The independent distribution has no sibling checkout")
        local = repository / "system"
        remote = other / "system"
        paths = set(path.relative_to(local) for path in (local / "flcore/security").rglob("*.py"))
        paths.update(path.relative_to(remote) for path in (remote / "flcore/security").rglob("*.py"))
        paths.update(path.relative_to(local) for path in (local / "tools").glob("*.py"))
        paths.update(Path(name) for name in (
            "flcore/servers/serverbase.py", "flcore/servers/serverfedsgd.py",
            "flcore/clients/clientbase.py", "flcore/clients/clientavg.py",
            "flcore/clients/clientfedsgd.py",
        ))
        for path in sorted(paths):
            with self.subTest(path=str(path)):
                self.assertEqual((local / path).read_text(encoding="utf-8-sig"),
                                 (remote / path).read_text(encoding="utf-8-sig"))
        main_files = [(root / "main.py").read_text(encoding="utf-8-sig") for root in (local, remote)]
        validators = []
        for source in main_files:
            validator = next(node for node in ast.parse(source).body
                             if isinstance(node, ast.FunctionDef) and node.name == "validate_security_options")
            validators.append(ast.dump(validator, include_attributes=False))
        self.assertEqual(*validators)
        arguments = [source[source.index("     # Attack configuration"):source.index("    args = parser.parse_args()")]
                     for source in main_files]
        self.assertEqual(*arguments)


if __name__ == "__main__":
    unittest.main()
