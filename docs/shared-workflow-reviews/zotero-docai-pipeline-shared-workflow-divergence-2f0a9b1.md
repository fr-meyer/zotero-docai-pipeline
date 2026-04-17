# Shared workflow divergence review for `fr-meyer/zotero-docai-pipeline`

Shared source: `fr-meyer/agent-toolkit` at `2f0a9b1049dd95212ffaad3467f30812d2c1f42b`
Target branch: `feature/package-cli-app`

## 1. Scope

Compared these registered shared starter templates against the current consumer workflow files:

- `templates/starter-workflows/coderabbit-pr-automation-pr-trigger.yml` -> `.github/workflows/coderabbit-pr-automation-pr.yml`
- `templates/starter-workflows/coderabbit-pr-automation-manual-trigger.yml` -> `.github/workflows/coderabbit-pr-automation-manual.yml`

## 2. Verified diff facts

### `.github/workflows/coderabbit-pr-automation-pr.yml`
- Pinned reusable-workflow refs differ: current=['74ed944bc852e9446f076223659733d18b5f8f96'], candidate=['74ed944bc852e9446f076223659733d18b5f8f96'].

### `.github/workflows/coderabbit-pr-automation-manual.yml`
- Pinned reusable-workflow refs differ: current=['74ed944bc852e9446f076223659733d18b5f8f96'], candidate=['74ed944bc852e9446f076223659733d18b5f8f96'].

## 3. Interpretation

The updater blocked a normal sync PR because these files do not exactly match the current or registered historical starter-template path lineage.
That does not prove the consumer is wrong. It means this case needs adjudication before an exact-managed overwrite is allowed.

## 4. Confidence and doubts

- Confidence: moderate, based on deterministic file comparison and contract-level fact extraction.
- Doubt: the remaining differences may reflect either older managed lineage outside the current exact path history, or an intentional consumer-specific operational choice.

## 5. Recommendation

Recommendation: adjudicate manually before any normalization PR is merged.

Questions to answer in review:
- Should this consumer remain on the older dynamic `shared_repository_ref` behavior, or be normalized to the current pinned shared-template model?
- Should `WORKFLOW_PUSH_TOKEN` passthrough be adopted here, or intentionally remain absent?
- Should this consumer continue forcing `auto_commit` / `auto_push`, or should it inherit the newer shared starter-template defaults?

Any proposed normalization patch in this PR is optional and should be treated as review material, not as an automatically approved overwrite.
