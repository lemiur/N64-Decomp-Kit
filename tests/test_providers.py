"""Model providers behind one protocol (``n64dk.match.providers``): dossiers in, proposals out with
identity and digests; mock, no-model and local-command adapters (a fake runner: nothing is
executed); credentials never stored; records compared across providers."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk.match import providers as pv
from n64dk.match import refine

ANSWER = "Here it is:\n```c\nint func_80001000(void) {\n    return 1;\n}\n```\n"


def request(name="func_80001000", prompt="## Target assembly\njr $ra\n"):
    return pv.dossier(name, "ido5.3 -O2", prompt, case=f"main:80001000:20", round_number=1)


class ProviderTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-providers-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_dossier_schema(self):
        d = request()
        self.assertEqual(pv.validate_dossier(d), [])
        self.assertEqual(pv.request_digest(d), pv.request_digest(request()))
        self.assertTrue(pv.validate_dossier({**d, "prompt": "changed"}))
        self.assertTrue(pv.validate_dossier({"schema": "other"}))

    def test_every_adapter_satisfies_the_protocol(self):
        fake = mock.Mock(return_value=ANSWER)
        for provider in (pv.NoModelProvider(), pv.MockProvider(),
                         pv.LocalCommandProvider(["fake-model-cli", "--model", "m"], model="m", runner=fake)):
            with self.subTest(provider=provider.identity.provider):
                self.assertIsInstance(provider, pv.Provider)
                proposal = provider.propose(request())
                self.assertEqual(proposal.request_digest, pv.request_digest(request()))
                self.assertIn("identity_digest", proposal.identity)

    def test_mock_is_deterministic_and_no_model_answers_nothing(self):
        a, b = pv.MockProvider().propose(request()), pv.MockProvider().propose(request())
        self.assertEqual(a.response_digest, b.response_digest)
        self.assertIn("int func_80001000(void)", a.candidate)
        fixed = pv.MockProvider({"func_80001000": ANSWER}).propose(request())
        self.assertEqual(fixed.candidate, "int func_80001000(void) {\n    return 1;\n}\n")
        none = pv.NoModelProvider().propose(request())
        self.assertEqual((none.error, none.candidate, none.response_digest), ("no model configured", None, None))
        bad = pv.MockProvider().propose({"schema": "x"})
        self.assertIn("dossier", bad.error)

    def test_local_command_passes_secrets_by_environment_only(self):
        seen = {}

        def runner(command, prompt, timeout, env, cwd=None, output_limit=0):
            seen.update(command=command, prompt=prompt, env=env, cwd=cwd)
            return ANSWER
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "sk-" + "a" * 40}):
            provider = pv.LocalCommandProvider(["fake-model-cli"], model="m1", version="2026-01",
                                               env_names=("MODEL_API_KEY",), runner=runner)
            proposal = provider.propose(request())
        self.assertEqual(seen["env"], {"MODEL_API_KEY": "sk-" + "a" * 40})   # reaches the command ...
        self.assertTrue(seen["cwd"])
        record = json.dumps(proposal.record())
        self.assertNotIn("sk-", record)                                       # ... and is stored nowhere
        self.assertEqual(proposal.identity["executable"], "fake-model-cli")
        self.assertEqual(proposal.candidate, "int func_80001000(void) {\n    return 1;\n}\n")
        for argv in (["cli", "--api-key", "abc123"], ["cli", "--token=abc"], ["cli", "sk-" + "b" * 30],
                     ["cli", "-H", "Bearer abcdefghijklmnop"]):
            with self.subTest(argv=argv), self.assertRaises(pv.CredentialLeak):
                pv.LocalCommandProvider(argv)
        pv.LocalCommandProvider(["cli", "--max-tokens", "100", "--model", "m"])        # not credentials

    def test_a_failing_command_is_an_error_proposal(self):
        def runner(*_args, **_kwargs):
            raise refine.RefineError("fake-model-cli exited 1")
        proposal = pv.LocalCommandProvider(["fake-model-cli"], runner=runner).propose(request())
        self.assertEqual((proposal.error, proposal.candidate), ("fake-model-cli exited 1", None))

    def test_records_never_hold_credentials(self):
        for value in ({"api_key": "x"}, {"nested": {"Authorization": "y"}}, {"note": "Bearer abcdefghijklmnopq"},
                      ["ghp_" + "c" * 30]):
            with self.subTest(value=value), self.assertRaises(pv.CredentialLeak):
                pv.assert_no_credentials(value)
        pv.assert_no_credentials({"author": "x", "max_tokens": 5, "env_names": ["MODEL_API_KEY"], "api_key": ""})

    def test_refine_runner_adapter(self):
        log = pv.ProviderLog(self.root / "providers.jsonl")
        run = pv.as_refine_runner(pv.MockProvider({"f": ANSWER}), log, name="f", recipe="ido5.3 -O2")
        self.assertEqual(run(["ignored"], "prompt text", 10.0, {}, cwd=str(self.root), output_limit=4096), ANSWER)
        with self.assertRaises(refine.RefineError):
            pv.as_refine_runner(pv.NoModelProvider(), log, name="f")(["ignored"], "prompt", 1.0)
        rows = log.read()
        self.assertEqual([r["error"] for r in rows], [None, "no model configured"])
        self.assertNotIn("prompt text", (self.root / "providers.jsonl").read_text())   # no prompt, no answer text

    def test_compare_providers_on_the_same_requests(self):
        log = pv.ProviderLog(self.root / "compare.jsonl")
        a = pv.MockProvider({"f": ANSWER}, model="a")
        b = pv.MockProvider(model="b")
        for name in ("f", "g"):
            for provider, score in ((a, 0 if name == "f" else 7), (b, 12)):
                proposal = provider.propose(request(name))
                log.append(proposal, name=name)
                log.outcome(proposal.request_digest, proposal.identity, score=score, exact=score == 0)
        result = pv.compare(log.read())
        self.assertEqual(result["requests"], 2)
        by_model = {p["model"]: p for p in result["providers"].values()}
        self.assertEqual((by_model["a"]["calls"], by_model["a"]["exact"], by_model["a"]["mean_score"]), (2, 1, 3.5))
        self.assertEqual(by_model["b"]["mean_score"], 12.0)
        (pair,) = result["pairs"].values()
        self.assertEqual(pair["shared"], 2)
        self.assertEqual(pair["first_better"] + pair["second_better"], 2)
        self.assertEqual(pair["identical"], 1)                           # both mocks answer "g" the same way


if __name__ == "__main__":
    unittest.main()
