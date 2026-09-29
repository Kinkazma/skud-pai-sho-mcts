"""Small gallery/HTTP/backfill tests; replay export is a mocked subprocess."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import paisho_game_gallery as gallery
import paisho_control as control


class GalleryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.examples = self.root / "examples"
        self.examples.mkdir()
        self.name = "generation-00000000000000000059"
        self.member = self.examples / self.name
        self.member.mkdir()
        (self.member / "metadata.json").write_text(json.dumps({"generation": 59, "opponent": "<script>bad()</script>"}))
        (self.member / "game.psr").write_text("PAISHO-RECORD 1\nexample\n")
        self.reader = gallery.Gallery(self.examples)

    def test_archive_preserves_colliding_generations_and_discovers_new_segments(self):
        from shutil import copytree
        first = self.root / "teacher-program-005/warmup-ppo/highlights"
        second = self.root / "teacher-program-006/warmup-ppo/highlights"
        copytree(self.examples, first)
        copytree(self.examples, second)
        (second / self.name / "game.psr").write_text("second campaign PSR")
        archive = gallery.ArchiveGallery(self.root)
        sources = archive.sources()
        self.assertEqual(len(sources), 2)
        with patch.object(gallery.subprocess, "run", side_effect=AssertionError("no exporter")):
            for key, directory in sources.items():
                page = archive.response("/examples/archive/" + key)
                self.assertEqual(page[0], 200)
                url = "/examples/archive/" + key + "/files/" + self.name + ".psr"
                self.assertIn(url, page[2].decode())
                result = archive.response(url)
                self.assertEqual(result[2], (directory / self.name / "game.psr").read_bytes())
        copytree(self.examples, self.root / "teacher-program-006/occurrence-000001/ppo/highlights")
        self.assertEqual(len(archive.sources()), 3)
        self.assertEqual(archive.response("/examples/archive/../../secret")[0], 404)

    def test_archive_root_is_recent_first_table_without_directory_links(self):
        import os
        from shutil import copytree
        first = self.root / "teacher-program-005/warmup-ppo/highlights"
        second = self.root / "teacher-program-006/warmup-ppo/highlights"
        copytree(self.examples, first)
        copytree(self.examples, second)
        os.utime(first / self.name / "game.psr", ns=(100, 100))
        os.utime(second / self.name / "game.psr", ns=(300, 300))
        old = first / "generation-270"
        old.mkdir()
        (old / "metadata.json").write_text(json.dumps({"generation": 270}))
        (old / "game.psr").write_text("older high generation")
        os.utime(old / "game.psr", ns=(200, 200))
        reader = gallery.ArchiveGallery(self.root)
        page = reader.response("/examples")[2].decode()
        self.assertIn("<table>", page)
        self.assertIn("Copier le texte PSR", page)
        self.assertNotIn("<ul>", page)
        self.assertNotIn("warmup-ppo", page)
        self.assertEqual(page.count("Collecte G59"), 1)
        self.assertLess(page.index("Collecte G59"), page.index("Collecte G270"))
        key = next(k for k, p in reader.sources().items() if p == second)
        self.assertIn("/examples/archive/" + key + "/files/" + self.name + ".psr", page)

    def test_page_reads_metadata_only_and_offers_psr(self):
        original = gallery.read_member
        def guarded(root, generation, member, limit=None):
            self.assertIn(member, ("meta.json", "metadata.json"))
            return original(root, generation, member, limit)
        with patch.object(gallery, "read_member", side_effect=guarded), patch.object(gallery.subprocess, "run", side_effect=AssertionError("no export on HTTP")):
            page = self.reader.page().decode()
        self.assertIn("Collecte G59", page)
        self.assertIn("avant son apprentissage", page)
        self.assertIn("Copier le texte PSR", page)
        self.assertIn("navigator.clipboard.writeText", page)
        self.assertNotIn("<script>bad()", page)
        self.assertNotIn('href="/examples/metadata.json', page)

    def test_download_is_exact_psr(self):
        status, content_type, data, filename = self.reader.response(f"/examples/files/{self.name}.psr")
        self.assertEqual(status, 200)
        self.assertEqual(data, (self.member / "game.psr").read_bytes())
        self.assertEqual(filename, self.name + ".psr")
        self.assertEqual(content_type, "text/plain; charset=utf-8")

    def test_host_and_guest_follow_recorded_neural_seat(self):
        for side, cells in [("host", "<td>Réseau neuronal</td><td>Bot aléatoire</td>"),
                            ("guest", "<td>Bot aléatoire</td><td>Réseau neuronal</td>")]:
            (self.member / "metadata.json").write_text(json.dumps({
                "generation": 59, "opponent": "random", "selected": {"neural_side": side}}))
            self.assertIn(cells, self.reader.page().decode())

    def test_traversal_and_json_not_served(self):
        for path in ("/examples/../secret", "/examples/files/%2e%2e/secret.psr",
                     "/examples/files/../../secret.psr", f"/examples/files/{self.name}.json",
                     f"/examples/{self.name}/metadata.json", "/examples/files//etc/passwd"):
            with self.subTest(path=path):
                self.assertEqual(self.reader.response(path)[0], 404)

    def test_winner_comes_from_terminal_result_not_selection_preference(self):
        for outcome, label in [("host-wins", "Hôte — Bot aléatoire"),
                               ("guest-wins", "Visiteur — Réseau neuronal"),
                               ("draw", "Partie nulle"), (None, "Non renseigné")]:
            (self.member / "metadata.json").write_text(json.dumps({
                "generation": 59, "opponent": "random", "selected": {
                    "neural_side": "guest", "terminal_outcome": outcome}}))
            page = self.reader.page().decode()
            self.assertIn("<th>Gagnant</th>", page)
            self.assertIn(f"<td>{label}</td>", page)

    def test_symlink_files_and_directories_rejected(self):
        secret = self.root / "secret.psr"
        secret.write_text("secret")
        (self.member / "game.psr").unlink()
        (self.member / "game.psr").symlink_to(secret)
        self.assertEqual(self.reader.entries(), [])
        self.assertEqual(self.reader.response(f"/examples/files/{self.name}.psr")[0], 404)
        other = self.examples / "generation-00000000000000000060"
        other.symlink_to(self.member, target_is_directory=True)
        self.assertEqual(self.reader.response(f"/examples/files/{other.name}.psr")[0], 404)

    def test_disabled_missing_and_partial_gallery(self):
        self.assertIn("Aucun dossier", gallery.Gallery(None).page().decode())
        (self.member / "metadata.json").unlink()
        self.assertEqual(self.reader.entries(), [])
        self.assertEqual(self.reader.response(f"/examples/files/{self.name}.psr")[0], 404)

    def test_controller_optional_config_and_http(self):
        raw = {"html_path": str(self.root / "dashboard.html")}
        self.assertIsNone(control._parse_dashboard(raw).examples_directory)
        raw["examples_directory"] = str(self.examples)
        config = SimpleNamespace(dashboard=control._parse_dashboard(raw), control_token="test")
        supervisor = SimpleNamespace(config=config)
        server = ThreadingHTTPServer(("127.0.0.1", 0), control.make_handler(supervisor, None))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", f"/examples/files/{self.name}.psr")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("attachment", response.getheader("Content-Disposition"))
            self.assertEqual(response.read(), (self.member / "game.psr").read_bytes())
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_flat_live_layout_and_numeric_metadata(self):
        metadata = self.examples / ".metadata"
        metadata.mkdir()
        for generation, filename in ((60, "generation-60.json"), (61, "61.json")):
            (metadata / filename).write_text(json.dumps({"generation": generation, "selected": {"game_id": 1}}))
            (self.examples / f"generation-{generation}.psr").write_text(f"PSR{generation}")
            result = self.reader.response(f"/examples/files/generation-{generation}.psr")
            self.assertEqual(result[0], 200)
            self.assertEqual(result[2], f"PSR{generation}".encode())
        self.assertEqual([meta["generation"] for _, meta in self.reader.entries()], [61, 60, 59])
        (metadata / "generation-62.json").write_text(json.dumps({"generation": 62, "selected": None}))
        self.assertIn("Collecte G62", self.reader.page().decode())
        self.assertEqual(self.reader.response("/examples/files/generation-62.psr")[0], 404)

    def test_background_exports_have_independent_clock_not_on_request_thread(self):
        raw = {"html_path": str(self.root / "dashboard.html"), "refresh_command": ["fake-dashboard"],
               "examples_directory": str(self.examples), "examples_campaign_directory": str(self.root / "campaign"),
               "examples_exporter": str(self.root / "exporter")}
        provider = control.DashboardProvider(control._parse_dashboard(raw))
        order = []
        with patch.object(control.subprocess, "run", side_effect=lambda *a, **k: order.append("dashboard")), \
             patch.object(control, "backfill_examples", side_effect=lambda *a, **k: order.append(("export", k["jobs"]))):
            provider._refresh()
            self.assertEqual(order, ["dashboard"])
            provider._refresh_examples()
        self.assertEqual(order, ["dashboard", ("export", 1)])
        provider._last_examples_refresh = 0
        with patch.object(control.threading, "Thread") as thread, patch.object(control, "backfill_examples") as export:
            provider.request_refresh()
            provider.request_refresh()
        thread.assert_called_once()
        export.assert_not_called()

    def test_background_refresh_optional_fields_default_none(self):
        config = control._parse_dashboard({"html_path": str(self.root / "dashboard.html")})
        self.assertIsNone(config.examples_campaign_directory)
        self.assertIsNone(config.examples_exporter)
        with patch.object(control, "backfill_examples") as export:
            control.DashboardProvider(config)._refresh()
        export.assert_not_called()


class BackfillTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.campaign = self.root / "stable"
        self.examples = self.root / "examples"
        self.calls = []
        self.template = "fake-export"
        self.make_generation(59)
        self.make_generation(60, complete=False)

    def test_paused_campaign_defers_exports_without_starting_a_process(self):
        with patch.object(gallery.subprocess, "run", side_effect=AssertionError("paused")):
            result = gallery.backfill(self.campaign, self.examples, self.template, allowed=lambda: False)
        self.assertEqual(result, [{"generation": 59, "status": "deferred_while_paused"}])

    def make_generation(self, generation, complete=True):
        directory = self.campaign / "generations" / f"generation-{generation:020}"
        actors = directory / "actors" / "committed"
        actors.mkdir(parents=True)
        snapshot = actors / "snapshot.psrsnap"
        snapshot.write_text("not read by gallery")
        abandoned = directory / "actors" / "abandoned"
        abandoned.mkdir()
        (abandoned / "snapshot.psrsnap").write_text("do not export")
        for name, format_, payload in (
            ("plan", "paisho-generation-plan-v1", {"generation": generation, "actor": {"opponent": "random"}}),
            ("actor-stage", "paisho-generation-actor-stage-v1", {"snapshot": str(snapshot.relative_to(self.campaign)), "behavior_producer": "producer", "snapshot_sha256": "digest"}),
            ("outcome", "paisho-generation-outcome-v1", {"generation": generation})):
            if name == "outcome" and not complete:
                continue
            (directory / (name + ".json")).write_text(json.dumps({"format": format_, "payload": payload}))

    def fake(self, argv, stdout, stderr, env, check):
        self.calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        generation = int(argv[argv.index("--generation") + 1])
        name = f"generation-{generation:020}"
        (output / name).mkdir()
        (output / name / "best-game.psr").write_text("PAISHO-RECORD 1\nselected\n")
        (output / name / "meta.json").write_text(json.dumps({
            "format": "paisho-local-neural-highlight-v1", "generation": generation,
            "neural_producer": "producer", "selected": {"game_id": 1}}))
        self.assertEqual(env["RAYON_NUM_THREADS"], "1")
        return subprocess.CompletedProcess(argv, 0)

    def test_committed_source_once_and_psr_publication(self):
        with patch.object(gallery.subprocess, "run", side_effect=self.fake):
            report = gallery.backfill(self.campaign, self.examples, self.template)
            repeat = gallery.backfill(self.campaign, self.examples, self.template)
        self.assertEqual(report, [{"generation": 59, "status": "exported"}])
        self.assertEqual(repeat, [{"generation": 59, "status": "existing"}])
        self.assertEqual(len(self.calls), 1)
        self.assertIn("/committed/", self.calls[0][2])
        self.assertEqual(len(gallery.Gallery(self.examples).entries()), 1)

    def test_targeted_generation_skips_other_completed(self):
        self.make_generation(61)
        with patch.object(gallery.subprocess, "run", side_effect=self.fake):
            report = gallery.backfill(self.campaign, self.examples, self.template, generations={59})
        self.assertEqual([row["generation"] for row in report], [59])

    def test_no_eligible_game_is_published_note_and_not_retried(self):
        def empty(argv, **kwargs):
            result = self.fake(argv, **kwargs)
            output = Path(argv[argv.index("--output") + 1])
            next(output.glob("*/best-game.psr")).unlink()
            path = next(output.glob("*/meta.json"))
            meta = json.loads(path.read_text())
            meta["selected"] = None
            path.write_text(json.dumps(meta))
            return result
        with patch.object(gallery.subprocess, "run", side_effect=empty):
            report = gallery.backfill(self.campaign, self.examples, self.template)
            self.assertEqual(report[0]["status"], "no_eligible_psr")
            gallery.backfill(self.campaign, self.examples, self.template)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Aucune partie neuronale compatible", gallery.Gallery(self.examples).page().decode())

    def test_completed_rust_output_recovered_without_reexport(self):
        rename = Path.rename
        def interrupted(path, destination):
            if path.name.startswith("publish-"):
                raise RuntimeError("helper publication interrupted")
            return rename(path, destination)
        with patch.object(gallery.subprocess, "run", side_effect=self.fake), patch.object(Path, "rename", interrupted):
            with self.assertRaises(RuntimeError):
                gallery.backfill(self.campaign, self.examples, self.template)
        with patch.object(gallery.subprocess, "run", side_effect=AssertionError("reuse completed Rust output")):
            self.assertEqual(gallery.backfill(self.campaign, self.examples, self.template)[0]["status"], "exported")
        self.assertEqual(len(self.calls), 1)

    def test_actual_direct_g59_contract(self):
        root = Path(__file__).resolve().parents[1] / "training-runs/game-examples-direct"
        name = "generation-00000000000000000059"
        psr = root / name / "best-game.psr"
        if not psr.exists():
            self.skipTest("local direct G59 export unavailable")
        reader = gallery.Gallery(root)
        self.assertTrue(reader.entry(name)["available"])
        self.assertIn("Collecte G59", reader.page().decode())
        response = reader.response(f"/examples/files/{name}.psr")
        self.assertEqual(response[0], 200)
        self.assertEqual(response[2], psr.read_bytes())

    def test_real_g59_metadata_only(self):
        campaign = Path(__file__).resolve().parents[1] / "training-runs/pure-curriculum-001/campaign"
        stage = campaign / "generations/generation-00000000000000000059/actor-stage.json"
        if not stage.exists():
            self.skipTest("local stable G59 archive unavailable")
        with patch.object(gallery.subprocess, "run", side_effect=AssertionError("metadata only")):
            job = next(job for job in gallery.completed_generations(campaign) if job["generation"] == 59)
        expected = json.loads(stage.read_text())["payload"]
        self.assertEqual(job["snapshot"], str((campaign / expected["snapshot"]).resolve()))
        self.assertEqual(job["producer"], expected["behavior_producer"])
        self.assertEqual(job["snapshot_sha256"], expected["snapshot_sha256"])

    def test_failed_export_preserved_retry_fresh(self):
        def fail(argv, **kwargs):
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir()
            (output / "partial").write_text("preserved")
            return subprocess.CompletedProcess(argv, 1)
        with patch.object(gallery.subprocess, "run", side_effect=fail), self.assertRaises(RuntimeError):
            gallery.backfill(self.campaign, self.examples, self.template)
        self.assertEqual(gallery.Gallery(self.examples).entries(), [])
        with patch.object(gallery.subprocess, "run", side_effect=self.fake):
            gallery.backfill(self.campaign, self.examples, self.template)
        self.assertEqual(len(list((self.examples / ".exports").glob("*/output/partial"))), 1)


if __name__ == "__main__":
    unittest.main()
