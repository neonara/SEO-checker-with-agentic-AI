from agent import memory


class TestDomainOf:
    def test_extracts_netloc_from_full_url(self):
        assert memory.domain_of("https://example.com/path?x=1") == "example.com"

    def test_adds_scheme_when_missing(self):
        assert memory.domain_of("example.com") == "example.com"

    def test_preserves_subdomain(self):
        assert memory.domain_of("https://blog.example.com") == "blog.example.com"


class TestSaveAndRetrieveAudits:
    def test_save_then_get_last_audit_round_trips(self, tmp_db_path):
        report = {"overall_score": 88, "grade": "B", "url": "https://example.com", "categories": []}
        row_id = memory.save_audit("https://example.com", report)
        assert row_id == 1

        fetched = memory.get_last_audit("https://example.com")
        assert fetched["overall_score"] == 88
        assert fetched["grade"] == "B"
        assert "_timestamp" in fetched

    def test_get_last_audit_returns_none_for_unknown_domain(self, tmp_db_path):
        assert memory.get_last_audit("https://never-audited.example.com") is None

    def test_get_last_audit_returns_most_recent(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 60, "grade": "D"})
        memory.save_audit("https://example.com", {"overall_score": 75, "grade": "C"})
        fetched = memory.get_last_audit("https://example.com")
        assert fetched["overall_score"] == 75

    def test_history_scoped_by_domain(self, tmp_db_path):
        memory.save_audit("https://a.com", {"overall_score": 50, "grade": "F"})
        memory.save_audit("https://b.com", {"overall_score": 90, "grade": "A"})
        history_a = memory.get_history("https://a.com")
        assert len(history_a) == 1
        assert history_a[0]["overall_score"] == 50

    def test_history_respects_limit_and_ordering(self, tmp_db_path):
        for score in [60, 65, 70, 75, 80]:
            memory.save_audit("https://example.com", {"overall_score": score, "grade": "C"})
        history = memory.get_history("https://example.com", limit=2)
        assert len(history) == 2
        # Most recent first.
        assert history[0]["overall_score"] == 80
        assert history[1]["overall_score"] == 75

    def test_history_empty_for_unknown_domain(self, tmp_db_path):
        assert memory.get_history("https://nothing-here.example.com") == []

    def test_www_and_non_www_are_different_domains(self, tmp_db_path):
        memory.save_audit("https://www.example.com", {"overall_score": 70, "grade": "C"})
        assert memory.get_last_audit("https://example.com") is None
        assert memory.get_last_audit("https://www.example.com") is not None


class TestGetAllFullAudits:
    def test_returns_full_parsed_reports_with_metadata_merged_in(self, tmp_db_path):
        memory.save_audit("https://example.com", {
            "overall_score": 80, "grade": "B", "url": "https://example.com", "categories": [],
        })
        reports = memory.get_all_full_audits()
        assert len(reports) == 1
        assert reports[0]["overall_score"] == 80
        assert reports[0]["_domain"] == "example.com"
        assert "_id" in reports[0]
        assert "_timestamp" in reports[0]

    def test_returns_across_all_domains_not_just_one(self, tmp_db_path):
        memory.save_audit("https://a.com", {"overall_score": 50, "grade": "F"})
        memory.save_audit("https://b.com", {"overall_score": 90, "grade": "A"})
        reports = memory.get_all_full_audits()
        assert len(reports) == 2

    def test_empty_history_returns_empty_list(self, tmp_db_path):
        assert memory.get_all_full_audits() == []

    def test_respects_limit(self, tmp_db_path):
        for score in [60, 70, 80, 90]:
            memory.save_audit("https://example.com", {"overall_score": score, "grade": "C"})
        reports = memory.get_all_full_audits(limit=2)
        assert len(reports) == 2

    def test_skips_corrupted_row_instead_of_crashing(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 80, "grade": "B"})
        # Manually corrupt the stored report_json to simulate a bad row.
        with memory._connect() as conn:
            conn.execute("UPDATE audits SET report_json = 'not valid json' WHERE id = 1")
        reports = memory.get_all_full_audits()
        assert reports == []  # corrupted row skipped, not raised

class TestTrendBaselineSelection:
    """get_last_audit used to return the newest audit of the *domain*, so a
    scan of /blog was "trended" against the homepage, and a run that had lost
    half its categories became the baseline for the next one."""

    def test_a_different_page_of_the_same_domain_is_not_the_baseline(self, tmp_db_path):
        memory.save_audit("https://example.com/", {"overall_score": 90, "grade": "A"})
        memory.save_audit("https://example.com/blog", {"overall_score": 40, "grade": "F"})
        assert memory.get_last_audit("https://example.com")["overall_score"] == 90
        assert memory.get_last_audit("https://example.com/blog/")["overall_score"] == 40
        assert memory.get_last_audit("https://example.com/pricing") is None

    def test_scheme_and_trailing_slash_do_not_make_a_different_page(self, tmp_db_path):
        memory.save_audit("http://example.com/", {"overall_score": 70, "grade": "C"})
        assert memory.get_last_audit("https://example.com")["overall_score"] == 70

    def test_an_audit_that_skipped_checks_is_not_a_baseline(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 81, "grade": "B"})
        memory.save_audit("https://example.com", {
            "overall_score": 55, "grade": "F",
            "skipped_categories": [{"name": "Page Speed", "reason": "The check failed to complete."}],
        })
        assert memory.get_last_audit("https://example.com")["overall_score"] == 81


class TestPublicId:
    def test_saved_audit_is_retrievable_by_its_public_id(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 88, "grade": "B"}, public_id="abc123")
        fetched = memory.get_audit_by_public_id("abc123")
        assert fetched["overall_score"] == 88
        assert fetched["_stored_url"] == "https://example.com"
        assert "_timestamp" in fetched
        assert memory.get_audit_by_public_id("nope") is None

    def test_a_public_id_is_generated_when_none_is_given(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 88, "grade": "B"})
        public_id = memory.get_history("https://example.com")[0]["public_id"]
        assert len(public_id) == 32
        assert memory.get_audit_by_public_id(public_id)["overall_score"] == 88

    def test_database_created_before_public_ids_is_migrated_in_place(self, tmp_db_path):
        """The live volume and the seed database predate the column."""
        import sqlite3
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("CREATE TABLE audits (id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL, "
                     "url TEXT NOT NULL, timestamp TEXT NOT NULL, overall_score REAL NOT NULL, grade TEXT, "
                     "report_json TEXT NOT NULL)")
        for score in (60, 70):
            conn.execute("INSERT INTO audits (domain, url, timestamp, overall_score, grade, report_json) "
                         "VALUES ('example.com', 'https://example.com', '2026-09-01T00:00:00+00:00', ?, 'C', ?)",
                         (score, '{"overall_score": %d}' % score))
        conn.commit()
        conn.close()

        rows = memory.get_history("https://example.com")
        ids = [r["public_id"] for r in rows]
        assert all(ids) and len(set(ids)) == 2
        assert memory.get_audit_by_public_id(ids[0])["overall_score"] == 70
        # Stable across later calls: the backfill only touches empty rows.
        assert [r["public_id"] for r in memory.get_history("https://example.com")] == ids
