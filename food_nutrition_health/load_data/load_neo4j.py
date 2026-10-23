"""Load the 50-row samples in load_50/ into Neo4j as a food/nutrition graph.

Usage (from food_nutrition_health/):
    uv run .\\load_data\\load_neo4j.py

Connection details are read from ../workshop-hackathon/.env (NEO4J_URI, NEO4J_USERNAME,
NEO4J_PASSWORD, NEO4J_DATABASE). Every write uses MERGE, so re-running is safe.
"""
from pathlib import Path
import math

import pandas as pd
import truststore
from neo4j import GraphDatabase

# Verify TLS against the Windows certificate store, which trusts local TLS-inspecting
# proxies/antivirus that Python's bundled CA list does not
truststore.inject_into_ssl()

HERE = Path(__file__).resolve().parent
DATA = HERE / "load_50"
ENV_FILE = HERE.parents[1] / "workshop-hackathon" / ".env"


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def rows(name: str) -> list[dict]:
    """Read a sample CSV as strings; empty cells become None (Cypher converts types)."""
    df = pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
    return [
        {k: (v if v != "" and not (isinstance(v, float) and math.isnan(v)) else None) for k, v in r.items()}
        for r in df.to_dict("records")
    ]


CONSTRAINTS = [
    "CREATE CONSTRAINT food_fdc IF NOT EXISTS FOR (n:Food) REQUIRE n.fdc_id IS UNIQUE",
    "CREATE CONSTRAINT food_category_id IF NOT EXISTS FOR (n:FoodCategory) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT nutrient_id IF NOT EXISTS FOR (n:Nutrient) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (n:Brand) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT portion_id IF NOT EXISTS FOR (n:Portion) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT measure_unit_id IF NOT EXISTS FOR (n:MeasureUnit) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT wweia_code IF NOT EXISTS FOR (n:WweiaCategory) REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT ingredient_code IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (n:Component) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT product_code IF NOT EXISTS FOR (n:Product) REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (n:Country) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT product_category_name IF NOT EXISTS FOR (n:ProductCategory) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT fruit_name IF NOT EXISTS FOR (n:Fruit) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT produce_price_key IF NOT EXISTS FOR (n:ProducePrice) REQUIRE n.key IS UNIQUE",
]

# (csv file, Cypher run once with $rows)
LOADS = [
    ("usda_fdc__food_category.csv", """
        UNWIND $rows AS r
        MERGE (c:FoodCategory {id: toInteger(r.id)})
        SET c.code = r.code, c.description = r.description"""),
    ("usda_fdc__nutrient.csv", """
        UNWIND $rows AS r
        MERGE (n:Nutrient {id: toInteger(r.id)})
        SET n.name = r.name, n.unit = r.unit_name,
            n.nutrient_nbr = r.nutrient_nbr, n.rank = toFloat(r.rank)"""),
    ("usda_fdc__measure_unit.csv", """
        UNWIND $rows AS r
        MERGE (u:MeasureUnit {id: toInteger(r.id)})
        SET u.name = r.name"""),
    ("usda_fdc__wweia_food_category.csv", """
        UNWIND $rows AS r
        MERGE (w:WweiaCategory {code: toInteger(r.wweia_food_category)})
        SET w.description = r.wweia_food_category_description"""),
    ("usda_fdc__food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        SET f.description = r.description, f.data_type = r.data_type,
            f.publication_date = date(r.publication_date)
        WITH f, r WHERE r.food_category_id IS NOT NULL
        MERGE (c:FoodCategory {id: toInteger(r.food_category_id)})
        MERGE (f)-[:IN_CATEGORY]->(c)"""),
    ("usda_fdc__food_nutrient.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        MERGE (n:Nutrient {id: toInteger(r.nutrient_id)})
        MERGE (f)-[h:HAS_NUTRIENT]->(n)
        SET h.amount = toFloat(r.amount), h.data_points = toInteger(r.data_points),
            h.min = toFloat(r.min), h.max = toFloat(r.max), h.median = toFloat(r.median)"""),
    ("usda_fdc__branded_food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        SET f:BrandedFood, f.gtin_upc = r.gtin_upc, f.ingredients = r.ingredients,
            f.serving_size = toFloat(r.serving_size), f.serving_size_unit = r.serving_size_unit,
            f.household_serving = r.household_serving_fulltext,
            f.branded_food_category = r.branded_food_category, f.market_country = r.market_country
        WITH f, r WHERE r.brand_owner IS NOT NULL
        MERGE (b:Brand {name: r.brand_owner})
        MERGE (f)-[:MADE_BY]->(b)"""),
    ("usda_fdc__food_portion.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        MERGE (p:Portion {id: toInteger(r.id)})
        SET p.amount = toFloat(r.amount), p.description = r.portion_description,
            p.modifier = r.modifier, p.gram_weight = toFloat(r.gram_weight)
        MERGE (f)-[:HAS_PORTION]->(p)
        MERGE (u:MeasureUnit {id: toInteger(r.measure_unit_id)})
        MERGE (p)-[:MEASURED_IN]->(u)"""),
    ("usda_fdc__survey_fndds_food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        SET f.food_code = r.food_code
        MERGE (w:WweiaCategory {code: toInteger(r.wweia_category_code)})
        MERGE (f)-[:IN_WWEIA_CATEGORY]->(w)"""),
    ("usda_fdc__input_food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        MERGE (i:Ingredient {code: r.sr_code})
        SET i.description = r.sr_description
        MERGE (f)-[h:HAS_INGREDIENT]->(i)
        SET h.seq_num = toInteger(r.seq_num), h.amount = toFloat(r.amount),
            h.unit = r.unit, h.gram_weight = toFloat(r.gram_weight)"""),
    ("usda_fdc__food_component.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        MERGE (c:Component {name: r.name})
        MERGE (f)-[h:HAS_COMPONENT]->(c)
        SET h.pct_weight = toFloat(r.pct_weight), h.gram_weight = toFloat(r.gram_weight),
            h.is_refuse = r.is_refuse = 'Y'"""),
    ("usda_fdc__sr_legacy_food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        SET f:SRLegacyFood, f.ndb_number = r.NDB_number"""),
    ("usda_fdc__foundation_food.csv", """
        UNWIND $rows AS r
        MERGE (f:Food {fdc_id: toInteger(r.fdc_id)})
        SET f:FoundationFood, f.ndb_number = r.NDB_number"""),
    ("openfoodfacts__en.openfoodfacts.org.products.csv", """
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
        FOREACH (b IN [x IN split(coalesce(r.brands, ''), ',') WHERE trim(x) <> '' | trim(x)] |
            MERGE (br:Brand {name: b}) MERGE (p)-[:MADE_BY]->(br))
        FOREACH (c IN [x IN split(coalesce(r.countries_en, ''), ',') WHERE trim(x) <> '' | trim(x)] |
            MERGE (co:Country {name: c}) MERGE (p)-[:SOLD_IN]->(co))
        FOREACH (c IN [x IN split(coalesce(r.categories_en, ''), ',') WHERE trim(x) <> '' | trim(x)] |
            MERGE (pc:ProductCategory {name: c}) MERGE (p)-[:IN_CATEGORY]->(pc))"""),
    ("fruit_veg_prices__Fruit Prices 2020.csv", """
        UNWIND $rows AS r
        MERGE (f:Fruit {name: r.Fruit})
        MERGE (pp:ProducePrice {key: r.Fruit + '|' + r.Form})
        SET pp.form = r.Form, pp.year = 2020,
            pp.retail_price = toFloat(r.RetailPrice), pp.retail_price_unit = r.RetailPriceUnit,
            pp.yield = toFloat(r.Yield), pp.cup_equivalent_size = toFloat(r.CupEquivalentSize),
            pp.cup_equivalent_unit = r.CupEquivalentUnit,
            pp.cup_equivalent_price = toFloat(r.CupEquivalentPrice)
        MERGE (f)-[:HAS_PRICE]->(pp)"""),
]


def main() -> None:
    env = read_env(ENV_FILE)
    db = env.get("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"])) as driver:
        driver.verify_connectivity()
        print(f"Connected to {env['NEO4J_URI']} (database: {db})")

        for stmt in CONSTRAINTS:
            driver.execute_query(stmt, database_=db)

        for file, cypher in LOADS:
            data = rows(file)
            summary = driver.execute_query(cypher, rows=data, database_=db).summary
            c = summary.counters
            print(f"{file}: {len(data)} rows -> +{c.nodes_created} nodes, +{c.relationships_created} rels")

        records, _, _ = driver.execute_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC", database_=db
        )
        print("\nNode counts:", ", ".join(f"{r['label']}={r['n']}" for r in records))


if __name__ == "__main__":
    main()
