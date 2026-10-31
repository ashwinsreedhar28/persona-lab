"""Documentation amendments keep sealed weight references intact and reject stale inputs."""
import copy
import hashlib
import unittest

from abliteration.storage import identity
from cloud.update_card import amend


class CardUpdateTests(unittest.TestCase):
    def fixture(self):
        old=b'---\nlicense: apache-2.0\nbase_model: openai/gpt-oss-20b\n---\n# Model\nExperimental.\n'
        spec=lambda b:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        core={'output_files':{'README.md':spec(old),'model.safetensors':spec(b'unchanged weight payload')},
              'distribution':{'local_manifest_sha256':'original-export'}}
        core['manifest_sha256']=identity(core)
        return core,old

    def test_atomic_documentation_seal_preserves_weights_and_history(self):
        core,old=self.fixture();snapshot=copy.deepcopy(core)
        new=old+b'User reports continued refusals; efficacy not established.\n'
        updated,marker=amend(core,core['manifest_sha256'],old,new,'a'*40)
        self.assertEqual(core,snapshot)
        self.assertEqual(updated['output_files']['model.safetensors'],core['output_files']['model.safetensors'])
        self.assertEqual(updated['output_files']['README.md']['sha256'],hashlib.sha256(new).hexdigest())
        digest=updated.pop('manifest_sha256');self.assertEqual(identity(updated),digest);self.assertEqual(marker.strip(),digest)
        self.assertEqual(updated['distribution']['documentation_updates'][0]['previous_revision'],'a'*40)

    def test_invalid_seal_stale_card_and_changed_license_fail_closed(self):
        core,old=self.fixture()
        for complete,card,new in [('wrong',old,old),(core['manifest_sha256'],old+b'changed',old),
                                  (core['manifest_sha256'],old,old.replace(b'apache-2.0',b'mit'))]:
            with self.subTest(complete=complete),self.assertRaises(ValueError):
                amend(core,complete,card,new,'a'*40)


if __name__=='__main__':unittest.main()
