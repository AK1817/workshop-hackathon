"""Load the full Kaggle datasets into Neo4j, trimmed to fit an AuraDB Free instance.

AuraDB Free caps a graph at 200k nodes and 400k relationships; all rows of every file would be
~1.4M nodes and 9M+ relationships (7.5M of them nutrient links). So this loads:
  - every USDA reference table (food categories, nutrients, WWEIA categories) and all fruit prices
  - every generic USDA food (SR Legacy, Foundation, Survey FNDDS) with its portions, ingredients,
    components, and HAS_NUTRIENT links to the CORE_NUTRIENTS
  - a seeded random sample of BRANDED_N USDA branded foods and OFF_N Open Food Facts products
Core nutrient amounts are also stored as properties on every Food (e.g. f.protein_g), which costs
no relationships, so branded foods can be compared on nutrition too.

Usage (from food_nutrition_health/):
    uv run .\\load_data\\load_full.py --dry-run   # print projected counts, touch nothing
    uv run .\\load_data\\load_full.py             # wipe the graph, then load

Data comes from the kagglehub cache (downloaded on first run). Connection details are read from
../workshop-hackathon/.env, as in load_neo4j.py.
"""
from pathlib import Path
import csv
import sys

import kagglehub
import pandas as pd
from neo4j import GraphDatabase

from load_neo4j import CONSTRAINTS, ENV_FILE, LOADS, read_env

NODE_LIMIT, REL_LIMIT = 200_000, 400_000  # AuraDB Free
BRANDED_N = 30_000
OFF_N = 30_000
SEED = 42
BATCH = 5_000

GENERIC_TYPES = {"sr_legacy_food", "foundation_food", "survey_fndds_food"}

# nutrient id -> Food property; earlier ids win when a food has several for the same property
CORE_NUTRIENTS = {
    1008: "energy_kcal", 2047: "energy_kcal", 2048: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1258: "saturated_fat_g",
    1005: "carbohydrates_g",
    2000: "sugars_g", 1063: "sugars_g",
    1079: "fiber_g",
    1093: "sodium_mg",
    1087: "calcium_mg",
    1089: "iron_mg",
    1092: "potassium_mg",
    1162: "vitamin_c_mg",
    1253: "cholesterol_mg",
}

OFF_COLS = [
    "code", "url", "product_name", "quantity", "brands", "countries_en", "main_category_en",
    "ingredients_text", "serving_size", "nutrition_grade_fr", "nutrition-score-fr_100g",
    "energy_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",
]

LOAD = dict(LOADS)  # reuse the sample loader's Cypher for tables whose columns are unchanged

FOOD = """
    UNWIND $rows AS r
    MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
    SET f.description = r.description, f.data_type = r.data_type,
        f.publication_date = date(r.publication_date), f.ndb_number = r.ndb_number
    SET f += r.nutrients
    FOREACH (_ IN CASE WHEN r.data_type = 'sr_legacy_food' THEN [1] ELSE [] END | SET f:SRLegacyFood)
    FOREACH (_ IN CASE WHEN r.data_type = 'foundation_food' THEN [1] ELSE [] END | SET f:FoundationFood)
    FOREACH (_ IN CASE WHEN r.data_type = 'survey_fndds_food' THEN [1] ELSE [] END | SET f:SurveyFood)"""
# No Food-[:IN_CATEGORY]->FoodCategory: in this dataset version SR Legacy and Foundation foods have
# no food_category_id, and survey foods hold WWEIA codes there (covered by IN_WWEIA_CATEGORY)

SURVEY = """
    UNWIND $rows AS r
    MATCH (f:Food {fdc_id: toInteger(r.fdc_id)})
    SET f.food_code = r.food_code
    WITH f, r
    MATCH (w:WweiaCategory {code: toInteger(r.wweia_category_code)})
    MERGE (f)-[:IN_WWEIA_CATEGORY]->(w)"""

HAS_NUTRIENT = """
    UNWIND $rows AS r
    MATCH (f:Food {fdc_id: toInteger(r.fdc_id)})
    MATCH (n:Nutrient {id: toInteger(r.nutrient_id)})
    MERGE (f)-[h:HAS_NUTRIENT]->(n)
    SET h.amount = toFloat(r.amount), h.data_points = toInteger(r.data_points),
        h.min = toFloat(r.min), h.max = toFloat(r.max), h.median = toFloat(r.median)"""

PORTION = """
    UNWIND $rows AS r
    MATCH (f:Food {fdc_id: toInteger(r.fdc_id)})
    MERGE (p:Portion {id: toInteger(r.id)})
    SET p.amount = toFloat(r.amount), p.unit = r.unit, p.description = r.portion_description,
        p.modifier = r.modifier, p.gram_weight = toFloat(r.gram_weight)
    MERGE (f)-[:HAS_PORTION]->(p)"""

INGREDIENT = LOAD["usda_fdc__input_food.csv"].replace(
    "MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})", "MATCH (f:Food {fdc_id: toInteger(r.fdc_id)})")

COMPONENT = LOAD["usda_fdc__food_component.csv"].replace(
    "MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})", "MATCH (f:Food {fdc_id: toInteger(r.fdc_id)})")

BRANDED = """
    UNWIND $rows AS r
    MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
    SET f:BrandedFood, f.description = r.description, f.data_type = 'branded_food',
        f.publication_date = date(r.publication_date),
        f.gtin_upc = r.gtin_upc, f.ingredients = r.ingredients,
        f.serving_size = toFloat(r.serving_size), f.serving_size_unit = r.serving_size_unit,
        f.household_serving = r.household_serving_fulltext,
        f.branded_food_category = r.branded_food_category, f.market_country = r.market_country
    SET f += r.nutrients
    WITH f, r WHERE r.brand_owner IS NOT NULL
    MERGE (b:Brand {name: r.brand_owner})
    MERGE (f)-[:MADE_BY]->(b)"""

PRODUCT = """
    UNWIND $rows AS r
    MERGE (p:Product {code: r.code})
    SET p.name = r.product_name, p.url = r.url, p.quantity = r.quantity,
        p.serving_size = r.serving_size, p.ingredients_text = r.ingredients_text,
        p.nutrition_grade = r.nutrition_grade_fr,
        p.nutrition_score = toInteger(r.`nutrition-score-fr_100g`),
        p.energy_100g = toFloat(r.energy_100g), p.fat_100g = toFloat(r.fat_100g),
        p.saturated_fat_100g = toFloat(r.`saturated-fat_100g`),
        p.carbohydrates_100g = toFloat(r.carbohydrates_100g), p.sugars_100g = toFloat(r.sugars_100g),
        p.fiber_100g = toFloat(r.fiber_100g), p.proteins_100g = toFloat(r.proteins_100g),
        p.salt_100g = toFloat(r.salt_100g), p.sodium_100g = toFloat(r.sodium_100g)
    FOREACH (b IN r.brand_list | MERGE (br:Brand {name: b}) MERGE (p)-[:MADE_BY]->(br))
    FOREACH (c IN r.country_list | MERGE (co:Country {name: c}) MERGE (p)-[:SOLD_IN]->(co))
    FOREACH (c IN CASE WHEN r.main_category_en IS NULL THEN [] ELSE [r.main_category_en] END |
        MERGE (pc:ProductCategory {name: c}) MERGE (p)-[:IN_CATEGORY]->(pc))"""


def read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], **kw)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with missing values as None (Cypher converts types)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def split_list(s) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if isinstance(s, str) else []


def build(usda: Path, off_file: Path, fruit_file: Path) -> tuple[list, dict, dict]:
    """Read and trim the source files. Returns (steps, projected nodes, projected rels)."""
    food = read(usda / "food.csv")
    generic = food[food.data_type.isin(GENERIC_TYPES)].copy()

    ndb = pd.concat([read(usda / f, usecols=["fdc_id", "NDB_number"])
                     for f in ("sr_legacy_food.csv", "foundation_food.csv")])
    generic = generic.merge(ndb.rename(columns={"NDB_number": "ndb_number"}), on="fdc_id", how="left")

    branded = read(usda / "branded_food.csv")
    branded = branded[branded.brand_owner.notna()].sample(BRANDED_N, random_state=SEED)
    branded = branded.merge(food[["fdc_id", "description", "publication_date"]], on="fdc_id", how="left")

    # core nutrients for the foods we keep, read in chunks (food_nutrient.csv has 7.5M rows)
    keep = set(generic.fdc_id) | set(branded.fdc_id)
    ids = {str(i) for i in CORE_NUTRIENTS}
    parts = [c[c.nutrient_id.isin(ids) & c.fdc_id.isin(keep)]
             for c in read(usda / "food_nutrient.csv",
                           usecols=["fdc_id", "nutrient_id", "amount", "data_points", "min", "max", "median"],
                           chunksize=1_000_000)]
    fn = pd.concat(parts)

    order = {str(i): n for n, i in enumerate(CORE_NUTRIENTS)}
    props = fn[fn.amount.notna()].assign(prop=fn.nutrient_id.map(lambda i: CORE_NUTRIENTS[int(i)]),
                                         rank=fn.nutrient_id.map(order))
    props = props.sort_values("rank").drop_duplicates(["fdc_id", "prop"])
    nutrient_maps = (props.assign(amount=props.amount.astype(float))
                     .groupby("fdc_id")[["prop", "amount"]]
                     .apply(lambda g: dict(zip(g.prop, g.amount))))
    for df in (generic, branded):
        df["nutrients"] = df.fdc_id.map(nutrient_maps).apply(lambda m: m if isinstance(m, dict) else {})

    generic_ids = set(generic.fdc_id)
    has_nutrient = fn[fn.fdc_id.isin(generic_ids)]

    units = read(usda / "measure_unit.csv").rename(columns={"id": "measure_unit_id", "name": "unit"})
    portion = read(usda / "food_portion.csv")
    portion = portion[portion.fdc_id.isin(generic_ids)].merge(units, on="measure_unit_id", how="left")
    survey = read(usda / "survey_fndds_food.csv")
    ingredient = read(usda / "input_food.csv")
    ingredient = ingredient[ingredient.fdc_id.isin(generic_ids) & ingredient.sr_code.notna()]
    component = read(usda / "food_component.csv")
    component = component[component.fdc_id.isin(generic_ids) & component.name.notna()]

    off = pd.read_csv(off_file, sep="\t", usecols=OFF_COLS, dtype=str, quoting=csv.QUOTE_NONE,
                      on_bad_lines="skip", keep_default_na=False, na_values=[""], encoding_errors="replace")
    off = off[off.code.notna() & off.product_name.notna() & off.nutrition_grade_fr.notna()]
    off = off.drop_duplicates("code").sample(OFF_N, random_state=SEED)
    off["brand_list"] = off.brands.map(split_list)
    off["country_list"] = off.countries_en.map(split_list)

    refs = {f: read(usda / f) for f in ("food_category.csv", "nutrient.csv", "wweia_food_category.csv")}
    fruit = read(fruit_file)

    brands = set(branded.brand_owner) | {b for bs in off.brand_list for b in bs}
    nodes = {
        "FoodCategory": len(refs["food_category.csv"]), "Nutrient": len(refs["nutrient.csv"]),
        "WweiaCategory": len(refs["wweia_food_category.csv"]),
        "Food (generic)": len(generic), "Food (branded)": len(branded),
        "Portion": portion.id.nunique(), "Ingredient": ingredient.sr_code.nunique(),
        "Component": component.name.nunique(), "Brand": len(brands),
        "Product": len(off), "Country": len({c for cs in off.country_list for c in cs}),
        "ProductCategory": off.main_category_en.nunique(),
        "Fruit": fruit.Fruit.nunique(), "ProducePrice": len(fruit),
    }
    rels = {
        "IN_WWEIA_CATEGORY": int(survey.fdc_id.isin(generic_ids).sum()),
        "HAS_NUTRIENT": len(has_nutrient), "HAS_PORTION": len(portion),
        "HAS_INGREDIENT": len(ingredient), "HAS_COMPONENT": len(component),
        "MADE_BY": len(branded) + sum(len(set(bs)) for bs in off.brand_list),
        "SOLD_IN": sum(len(set(cs)) for cs in off.country_list),
        "IN_CATEGORY (Product)": int(off.main_category_en.notna().sum()),
        "HAS_PRICE": len(fruit),
    }

    steps = [
        ("food categories", LOAD["usda_fdc__food_category.csv"], refs["food_category.csv"]),
        ("nutrients", LOAD["usda_fdc__nutrient.csv"], refs["nutrient.csv"]),
        ("WWEIA categories", LOAD["usda_fdc__wweia_food_category.csv"], refs["wweia_food_category.csv"]),
        ("generic foods", FOOD, generic),
        ("survey food WWEIA links", SURVEY, survey),
        ("HAS_NUTRIENT", HAS_NUTRIENT, has_nutrient),
        ("portions", PORTION, portion),
        ("ingredients", INGREDIENT, ingredient),
        ("components", COMPONENT, component),
        ("branded foods", BRANDED, branded),
        ("Open Food Facts products", PRODUCT, off),
        ("fruit prices", LOAD["fruit_veg_prices__Fruit Prices 2020.csv"], fruit),
    ]
    return steps, nodes, rels


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    usda = Path(kagglehub.dataset_download("stoicstatic/fooddata-central-nutrition-and-component-data"))
    off_file = Path(kagglehub.dataset_download("openfoodfacts/world-food-facts")) / "en.openfoodfacts.org.products.tsv"
    fruit_file = Path(kagglehub.dataset_download("vstacknocopyright/fruit-and-vegetable-prices")) / "Fruit Prices 2020.csv"

    steps, nodes, rels = build(usda, off_file, fruit_file)
    n_nodes, n_rels = sum(nodes.values()), sum(rels.values())
    print("Projected nodes:", ", ".join(f"{k}={v:,}" for k, v in nodes.items()))
    print("Projected rels: ", ", ".join(f"{k}={v:,}" for k, v in rels.items()))
    print(f"Total: {n_nodes:,} / {NODE_LIMIT:,} nodes, {n_rels:,} / {REL_LIMIT:,} relationships")
    if n_nodes > NODE_LIMIT or n_rels > REL_LIMIT:
        sys.exit("Over the AuraDB Free limits; lower BRANDED_N / OFF_N and re-run.")
    if dry_run:
        return

    env = read_env(ENV_FILE)
    db = env.get("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"])) as driver:
        driver.verify_connectivity()
        print(f"\nConnected to {env['NEO4J_URI']} (database: {db})")

        # start from an empty graph so nodes left over from the 50-row sample don't count toward the limits
        while driver.execute_query("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS n",
                                   database_=db).records[0]["n"]:
            pass
        for stmt in CONSTRAINTS:
            driver.execute_query(stmt, database_=db)

        for name, cypher, df in steps:
            data = records(df)
            for i in range(0, len(data), BATCH):
                driver.execute_query(cypher, rows=data[i:i + BATCH], database_=db)
                print(f"\r{name}: {min(i + BATCH, len(data)):,} / {len(data):,}", end="", flush=True)
            print()

        records_, _, _ = driver.execute_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC", database_=db
        )
        print("\nNode counts:", ", ".join(f"{r['label']}={r['n']:,}" for r in records_))
        totals = driver.execute_query(
            "CALL () { MATCH (n) RETURN count(n) AS nodes } CALL () { MATCH ()-[r]->() RETURN count(r) AS rels } "
            "RETURN nodes, rels", database_=db
        ).records[0]
        print(f"Total: {totals['nodes']:,} nodes, {totals['rels']:,} relationships")


if __name__ == "__main__":
    main()
