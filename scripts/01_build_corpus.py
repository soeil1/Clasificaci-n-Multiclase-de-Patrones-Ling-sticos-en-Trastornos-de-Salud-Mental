"""Une los CSV crudos en un solo corpus y lo guarda en data/processed."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("pipe", os.path.join(ROOT,"scripts","02_run_pipeline.py"))
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)

os.makedirs(os.path.join(ROOT,"data","processed"), exist_ok=True)
df = pipe.build_corpus()
out = os.path.join(ROOT,"data","processed","corpus.csv")
df[["text","label"]].to_csv(out, index=False)
print(f"Corpus guardado en {out}: {len(df)} docs | {df['label'].value_counts().to_dict()}")
