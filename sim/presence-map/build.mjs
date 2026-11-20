import { stripTypeScriptTypes } from "node:module";
import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";

await mkdir("dist/src", { recursive: true });
const html = await readFile("index.html", "utf8");
await writeFile("dist/index.html", html.replace("./src/app.ts", "./src/app.js"));
await copyFile("style.css", "dist/style.css");

for (const name of await readdir("src")) {
  if (!name.endsWith(".ts")) continue;
  const source = await readFile(`src/${name}`, "utf8");
  await writeFile(`dist/src/${name.replace(/\.ts$/, ".js")}`, stripTypeScriptTypes(source));
}
