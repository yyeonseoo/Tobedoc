# Research execution policy

- Work within the user's existing ChatGPT Plus allowance. Do not purchase credits, call paid model APIs, or create a paid dependency without explicit new authorization.
- Use the assistant for clinical/scientific reasoning, study design, interpreting ambiguous evidence, and review of important code or results.
- Use ordinary local code for downloads, document parsing, schema checks, transformations, hashes, experiment orchestration, descriptive statistics and tests. Do not ask an LLM to do a deterministic task that a short script can perform reliably.
- Write full machine outputs to local files; return concise counts, errors and targeted evidence spans to the assistant. Preserve raw evidence. Cache immutable inputs and rerun only changed steps.
- A local shell command does not make the assistant's inference local or remove subscription usage. A local LLM is a separate runtime and is not required for routine data processing.
- The installed Ollama Qwen model is a candidate for local tasks, not an automatically selected research backbone or clinical judge. Record the exact digest, quantization, template, settings and context policy before experimental use.
- Do not silently replace the original framework's backend with Qwen or report a prompt port as a replication. Keep generating, judging and independent clinical evaluation distinct.
- Do not download models, install packages, launch remote jobs or alter Ollama server configuration merely to satisfy a preference for local work. First use existing resources; request a decision when the change is material.
- Only the current Mac has been inspected. Do not assume access to or specifications of the user's Mac mini. Keep scripts portable and transfer only explicitly authorized, scoped inputs/outputs.
- Never fabricate missing generation outputs or expert ratings. Existing source checks remain distinct from behavioural findings.
