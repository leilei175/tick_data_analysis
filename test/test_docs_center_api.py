import unittest

from factor_dashboard.app import app


class TestDocsCenterAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_docs_index_with_tags_and_search_highlight(self):
        resp = self.client.get('/api/docs/index')
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload['status'], 'success')
        self.assertIn('docs', payload['data'])
        self.assertIn('tags', payload['data'])

        search_resp = self.client.get('/api/docs/index?q=因子')
        self.assertEqual(search_resp.status_code, 200)
        search_payload = search_resp.get_json()
        self.assertEqual(search_payload['status'], 'success')
        if search_payload['data']['docs']:
            first = search_payload['data']['docs'][0]
            self.assertIn('title', first)
            self.assertIn('snippet', first)

    def test_docs_content_returns_error_for_invalid_doc_id(self):
        resp = self.client.get('/api/docs/content/not_exists')
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload['status'], 'error')


if __name__ == "__main__":
    unittest.main()
