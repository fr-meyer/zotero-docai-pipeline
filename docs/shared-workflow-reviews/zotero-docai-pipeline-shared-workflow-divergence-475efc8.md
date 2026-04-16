# Shared workflow divergence review for `fr-meyer/zotero-docai-pipeline`

Shared source: `fr-meyer/agent-toolkit` at `475efc8f871c85f67c3c5c81df3a78820bc317d9`
Target branch: `feature/package-cli-app`

## 1. Scope

Compared these registered shared starter templates against the current consumer workflow files:

- `templates/starter-workflows/coderabbit-pr-comment-trigger.yml` -> `.github/workflows/coderabbit-pr-comment-trigger.yml`

## 2. Verified diff facts

### `.github/workflows/coderabbit-pr-comment-trigger.yml`
- Pinned reusable-workflow refs are identical: both `current` and `candidate` resolve to `cec0072f25df02c22b7732059caddbff68c0fada`.

## 3. Interpretation

The updater blocked a normal sync PR because these files do not exactly match the current or registered historical starter-template path lineage.
That does not prove the consumer is wrong. It means this case needs adjudication before an exact-managed overwrite is allowed.

## 4. Confidence and doubts

- Confidence: moderate, based on deterministic file comparison and contract-level fact extraction.
- Doubt: the remaining differences may reflect either older managed lineage outside the current exact path history, or an intentional consumer-specific operational choice.

## 5. Recommendation

Recommendation: adjudicate manually before any normalization PR is merged.

Questions to answer in review:
- `shared_repository_ref` is pinned to a full commit SHA (not a dynamic ref) in `.github/workflows/coderabbit-pr-comment-trigger.yml`. Aside from bumping that pin when you intentionally adopt a newer shared workflow, does this consumer need any other change from the current pinned shared-template wiring?
- `WORKFLOW_PUSH_TOKEN` is already passed through to the reusable workflow. Is any explicit override or different secret mapping required for this repository?
- `auto_commit` and `auto_push` default to `false` via `vars.CODERABBIT_AUTO_COMMIT` and `vars.CODERABBIT_AUTO_PUSH`. Should this repository explicitly opt either into `true`, or keep the conservative defaults?

Any proposed normalization patch in this PR is optional and should be treated as review material, not as an automatically approved overwrite.
