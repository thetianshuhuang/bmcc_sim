import bmcc
from tqdm import tqdm
from scipy.stats import poisson


INSPECT = (
    "./data/r=0.7,sym=False/n=500,d=2,k=3/"
    "ff02a9dc-a683-11e9-9834-4c34887dea79.npz"
)

ds = bmcc.GaussianMixture(INSPECT, load=True)

ds.plot_actual(kwargs_scatter={"marker": "."}, plot=True)


model = bmcc.GibbsMixtureModel(
    data=ds.data,
    component_model=bmcc.NormalWishart(df=2),
    mixture_model=bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, 3)),
    thinning=20
)

for i in tqdm(range(20000)):
    model.iter()


res = model.select_lstsq(burn_in=0)
res.evaluate(ds.assignments, oracle=ds.oracle, oracle_matrix=ds.oracle_matrix)

res.trace(plot=True)
res.clustering(kwargs_scatter={"marker": "."}, plot=True)

