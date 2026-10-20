import { code, useData } from '../data'

const PIPELINE = [
  { title: 'Start from the original', text: 'Llama 3.1 8B Instruct in 16 bit, measured once as the reference that every other model is compared with.', path: 'evaluate.py' },
  { title: 'Round the weights', text: 'Round to nearest with one ruler per row or per 128 weights, and GPTQ, all written from scratch.', path: 'quantization/naive_ptq.py' },
  { title: 'Store real 4 bit', text: 'Two 4 bit integers per byte plus their rulers. The model drops from 15 GB to about 5 GB and runs fully on the GPU.', path: 'quantization/packed.py' },
  { title: 'Repair', text: 'Keep the most fragile tables at 8 bit, and test retrieval and QLoRA on top of the rounded model.', path: 'quantization/mixed_sensitivity.py' },
  { title: 'Measure', text: 'Perplexity, distance from the original, same next word and a 412 question exam, with identical inputs for every model.', path: 'quantization/evaluate_checkpoint.py' },
]

const PRINCIPLES = [
  { title: 'Same inputs for every model', text: 'The same text and the same questions, scored by the same code, so differences come from the models alone.' },
  { title: 'Separate text for choosing and for reporting', text: 'Choices such as which layers to protect were made on validation text; every reported number comes from the test text and the exam.' },
  { title: 'Question by question comparison', text: 'Answers are compared one question at a time, counting how many were fixed and how many were broken, not only as totals.' },
  { title: 'Failures reported as measured', text: 'Settings were fixed before looking at exam results, and retrieval and QLoRA are reported even though they did not help.' },
]

export default function Method() {
  const { results } = useData()
  return (
    <section id="method" className="section wrap" aria-labelledby="method-title">
      <div className="section-head">
        <h2 id="method-title">How it was built</h2>
        <p className="lede">
          Everything runs on one {results.facts.gpu} graphics card with free tools. The models are built layer by layer from the saved weight files,
          so the 15 GB original never has to fit in memory at once.
        </p>
      </div>
      <ol className="pipeline">
        {PIPELINE.map((s, i) => (
          <li key={s.title}>
            <span className="n">{i + 1}</span>
            <h3>{s.title}</h3>
            <p>{s.text} <a href={code(s.path)}>Read the code</a></p>
          </li>
        ))}
      </ol>
      <div className="principles">
        {PRINCIPLES.map((p) => (
          <div key={p.title}><h3>{p.title}</h3><p>{p.text}</p></div>
        ))}
      </div>
    </section>
  )
}
