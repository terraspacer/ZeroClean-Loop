# ZeroEC prompt source recovery

Audited 2026-10-26 against ZeroEC commit `d76ef45509d7e0f9e7cc0fa98d445c06afbb157c`.

## Fidelity conclusion

The exact omitted ZeroEC system prompt has **not** been recovered. Three files with the missing names were recovered verbatim from the same author's official [ZeroDC repository](https://github.com/YangChen32768/ZeroDC), pinned to commit `29cde9e0dbcdde04c34bbc4c185c73b357817390`. These are inspectable author-source candidates, not proof of the contents originally intended by the pinned ZeroEC program.

Two of those files match already committed ZeroEC artifacts byte for byte. The recovered system prompt differs from the committed ZeroEC system prompt. Consequently, using the recovered ZeroDC files supports an explicitly labeled author-source reconstruction; it does not establish an exact replay of pinned ZeroEC.

| Recovered author artifact | Pinned ZeroEC comparison | Byte identity | SHA-256 of recovered artifact |
| --- | --- | --- | --- |
| `HumanMessage_EC.txt` | `prompt_templates/HumanMessage.txt` | Yes | `3695f49cb835a0d2d931cccd91f1a1ebdceeb91b9be5ac5e6eb6ca7ca0b8d3f4` |
| `general_examples_EC.txt` | `prompt_templates/examples.txt` | Yes | `b420e6160f2a8d92c56930f1d08bb4d7d94fefd05f5cf73d9bb727d7e23b4ee6` |
| `SystemMessage_EC.txt` | `prompt_templates/SystemMessage.txt` | No | `aecc5b7719a0da05b01d7d544b64ebd7ab10af5c61fae6a92010747a0ce66248` |

The recovered files and machine-readable provenance are in [`artifacts/recovered_prompts/zerodc_29cde9e0dbcd/`](../artifacts/recovered_prompts/zerodc_29cde9e0dbcd/). The files were saved without modification.

## What the pinned ZeroEC source establishes

The public [zero-shot correction program](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction_zero_shot.py#L1241-L1243) names:

- `prompt_templates/SystemMessage_EC.txt`
- `prompt_templates/HumanMessage_EC.txt`
- `prompt_templates/general_examples_EC.txt`

Its [prompt construction](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction_zero_shot.py#L1063-L1073) passes general examples and a blank `dataset_specific_examples` value. Its introductory comment describes retrieved tuples plus a small set of general examples, while excluding automatically generated correction reasoning and correction rules. Thus omitting the fixed examples is a deliberate stricter zero-shot adaptation, rather than the behavior specified by this public call site.

The committed `SystemMessage.txt` expects `specific_examples`, whereas the missing-file call site supplies `dataset_specific_examples`. A placeholder rename can make a reconstruction runnable, but cannot establish the unknown original system-prompt text.

## Recovered primary sources and compatibility

All three recovered files first appear in ZeroDC's [initial commit](https://github.com/YangChen32768/ZeroDC/commit/4e0420044bf786416393eb1bd09b937d452e792f), dated 2026-07-21, after the pinned ZeroEC commit. Their history therefore does not establish that they are the earlier omitted files.

- [ZeroDC `SystemMessage_EC.txt`](https://github.com/YangChen32768/ZeroDC/blob/29cde9e0dbcdde04c34bbc4c185c73b357817390/prompt_templates/SystemMessage_EC.txt) uses the `general_examples` and `dataset_specific_examples` placeholders expected by the pinned ZeroEC zero-shot call site. Its wording differs from pinned ZeroEC `SystemMessage.txt`, including an instruction against null corrections. No author statement identifying this file as the omitted ZeroEC file was found.
- [ZeroDC `HumanMessage_EC.txt`](https://github.com/YangChen32768/ZeroDC/blob/29cde9e0dbcdde04c34bbc4c185c73b357817390/prompt_templates/HumanMessage_EC.txt) has the expected `Dirty_Tuple`, `Erroneous_value`, `Relevant_clean_tuples`, and `format_instructions` placeholders. Its bytes equal pinned ZeroEC `HumanMessage.txt`.
- [ZeroDC `general_examples_EC.txt`](https://github.com/YangChen32768/ZeroDC/blob/29cde9e0dbcdde04c34bbc4c185c73b357817390/prompt_templates/general_examples_EC.txt) parses successfully with the upstream loader's `ast.literal_eval` approach as a list of four fixed examples. Its bytes equal pinned ZeroEC `examples.txt`.

ZeroDC's [correction loader](https://github.com/YangChen32768/ZeroDC/blob/29cde9e0dbcdde04c34bbc4c185c73b357817390/correction.py#L1270-L1272) uses the same three filenames. Compatible filenames and placeholders support using these as author-source candidates; they do not prove historical identity.

## Bounded primary-source audit

| Source checked | Observed result |
| --- | --- |
| [Official ZeroEC branches API](https://api.github.com/repos/YangChen32768/ZeroEC/branches?per_page=100) | Only `main`; head remains the pinned `d76ef455...` commit. |
| [Official ZeroEC tags API](https://api.github.com/repos/YangChen32768/ZeroEC/tags?per_page=100) | Empty. |
| [Official ZeroEC releases API](https://api.github.com/repos/YangChen32768/ZeroEC/releases?per_page=100) | Empty; no release assets to inspect. |
| [Official ZeroEC pull requests API](https://api.github.com/repos/YangChen32768/ZeroEC/pulls?state=all&per_page=100) | Empty. No alternate submitted prompt files. |
| [Pinned ZeroEC prompt directory](https://github.com/YangChen32768/ZeroEC/tree/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/prompt_templates) | Eighteen files; none of the three `_EC` paths. All eight local commits had already been audited for these names before this recovery task. |
| [Author GitHub repositories](https://api.github.com/users/YangChen32768/repos?per_page=100) | `ZeroEC`, `ZeroDC`, and `ZeroDC2`. ZeroDC yielded the candidate artifacts; ZeroDC2's tree request returned HTTP 409 (empty repository). |
| [Coauthor Wei Ni's official homepage](https://welkinni.github.io/) | Lists the ZeroEC paper, but provides no PDF or supplement link for it. |
| [Coauthor Xiaoye Miao's official university homepage](https://person.zju.edu.cn/en/miaoxy_en) | Public rendered page exposed no ZeroEC prompt, paper, or supplement link. |
| [IEEE publisher record](https://ieeexplore.ieee.org/document/11113193/) and [DOI](https://doi.org/10.1109/ICDE65448.2025.00221) | Identified the official paper. The publisher page was robot-gated in the browsing tool. Direct publisher PDF/stamp requests returned HTTP 418; the media endpoint yielded an empty response. The Crossref-listed publisher PDF endpoint also returned HTTP 418. No full-text prompt appendix could be inspected through these accessible endpoints. This is an access limitation, not evidence that an appendix does not exist. |

Searches for the exact missing filenames and for ZeroEC paper/supplement/appendix material did not identify another primary source. No issues were posted and no author was contacted. The audit is intentionally bounded rather than treating web-search absence as proof of nonexistence.

## Remaining blocker

An exact pinned ZeroEC replay requires an authoritative copy of the original `SystemMessage_EC.txt`, or an author statement tying a recoverable file or transformation to that omitted path. The checked primary sources provide neither. An exact original set of all three missing files also remains a historical claim beyond the observed two byte matches.

No benchmark clean/reference data, detection masks, or correction outputs were opened for this audit. Fixed public prompt examples were inspected solely as the requested source artifacts. No paper performance figures were used or included. No model code was changed.
