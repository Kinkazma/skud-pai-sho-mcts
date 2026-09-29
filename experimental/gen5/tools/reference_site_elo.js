#!/usr/bin/env node
"use strict";

// Differential oracle for the pinned public Garden Gate Elo implementation.
// It reads official JavaScript and a TSV history, prints results, and never
// writes repository files.

const crypto = require("crypto");
const fs = require("fs");
const vm = require("vm");

const EXPECTED_SHA256 = "7976112f27b9e227ed80e2c7a7eaabf5cef0fbc79d05bf5459c0677e4fe950a0";

if (process.argv.length !== 5) {
  console.error("usage: reference_site_elo.js OFFICIAL_ELO.js GAMES.tsv INITIAL_RATING");
  process.exit(2);
}

const source = fs.readFileSync(process.argv[2], "utf8");
const digest = crypto.createHash("sha256").update(source).digest("hex");
if (digest !== EXPECTED_SHA256) {
  throw new Error(`official Elo source SHA-256 mismatch: ${digest}`);
}
const initialRating = Number.parseInt(process.argv[4], 10);
if (!Number.isSafeInteger(initialRating)) {
  throw new Error("initial rating must be a safe integer");
}

const context = { window: {} };
vm.runInNewContext(source, context, { filename: process.argv[2] });
const elo = context.window.Elo;
const lines = fs.readFileSync(process.argv[3], "utf8").trim().split(/\r?\n/);
const header = lines.shift().split("\t");
const column = Object.fromEntries(header.map((name, index) => [name, index]));
const games = lines.map((line) => {
  const fields = line.split("\t");
  return {
    sequence: Number.parseInt(fields[column.sequence], 10),
    pairId: Number.parseInt(fields[column.pair_id], 10),
    host: fields[column.host],
    guest: fields[column.guest],
    outcome: fields[column.outcome],
  };
});
games.sort((left, right) => left.sequence - right.sequence);

const agents = [...new Set(games.flatMap((game) => [game.host, game.guest]))].sort();
const ratings = new Map(agents.map((agent) => [agent, initialRating]));
console.log("PAISHO-SITE-ELO-INDEPENDENT-REFERENCE\t1");
for (const game of games) {
  const hostBefore = ratings.get(game.host);
  const guestBefore = ratings.get(game.guest);
  const hostScore = { H: 1, D: 0.5, G: 0 }[game.outcome];
  if (hostScore === undefined) {
    throw new Error(`unknown outcome ${game.outcome}`);
  }
  const updated = elo.getNewPlayerRatings(hostBefore, guestBefore, hostScore);
  const delta = updated.hostRating - hostBefore;
  ratings.set(game.host, updated.hostRating);
  ratings.set(game.guest, updated.guestRating);
  console.log(
    [
      "update",
      game.sequence,
      game.pairId,
      game.host,
      game.guest,
      hostBefore,
      guestBefore,
      delta,
      updated.hostRating,
      updated.guestRating,
    ].join("\t"),
  );
}
for (const agent of agents) {
  console.log(["rating", agent, ratings.get(agent)].join("\t"));
}
