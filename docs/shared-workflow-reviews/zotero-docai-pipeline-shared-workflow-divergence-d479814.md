# Shared workflow divergence review for `fr-meyer/zotero-docai-pipeline`

Shared source: `fr-meyer/agent-toolkit` at `d479814e962081efd87430626eb9c5e6a75ebe60`
Target branch: `feature/package-cli-app`

## 1. Scope

Compared these registered shared starter templates against the current consumer workflow files:

- `templates/starter-workflows/coderabbit-pr-automation-wrapper.yml` -> `.github/workflows/coderabbit-pr-automation.yml`
- `templates/starter-workflows/coderabbit-pr-comment-trigger.yml` -> `.github/workflows/coderabbit-pr-comment-trigger.yml`

## 2. Verified diff facts

### `.github/workflows/coderabbit-pr-automation.yml`
- Pinned reusable-workflow refs differ: current=['39a84b813769663a445cdeac62322ffd2ee8a435'], candidate=['743e51edf2385216507358e3f7fa285a318965d8'].
- Consumer file currently forces `auto_commit: true`, while the shared starter template defers to vars/default false.
- Consumer file currently forces `auto_push: true`, while the shared starter template defers to vars/default false.
- `shared_repository_ref` handling differs: current=['shared_repository_ref:', 'shared_repository_ref: ${{ inputs.shared_repository_ref }}', 'shared_repository_ref: main'], candidate=['shared_repository_ref: 743e51edf2385216507358e3f7fa285a318965d8', 'shared_repository_ref: 743e51edf2385216507358e3f7fa285a318965d8'].
- The input contract around `shared_repository_ref` differs between the consumer workflow and the shared starter template.

### `.github/workflows/coderabbit-pr-comment-trigger.yml`
- Pinned reusable-workflow refs differ: current=['39a84b813769663a445cdeac62322ffd2ee8a435'], candidate=['743e51edf2385216507358e3f7fa285a318965d8'].
- Consumer file currently forces `auto_commit: true`, while the shared starter template defers to vars/default false.
- Consumer file currently forces `auto_push: true`, while the shared starter template defers to vars/default false.
- `shared_repository_ref` handling differs: current=['shared_repository_ref: main'], candidate=['shared_repository_ref: 743e51edf2385216507358e3f7fa285a318965d8'].

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
- Should `WORKFLOW_PUSH_TOKEN` passthrough be retained/required here, or intentionally omitted?
- Should this consumer continue forcing `auto_commit` / `auto_push`, or should it inherit the newer shared starter-template defaults?

Any proposed normalization patch in this PR is optional and should be treated as review material, not as an automatically approved overwrite.
