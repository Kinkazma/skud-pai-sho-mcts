#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const { execFileSync } = require("node:child_process");

const EXPECTED_COMMIT = "b849dbdabb1138ff0f6d609adf38b301c2f875ae";
const referenceRoot = process.argv[2];
if (!referenceRoot) {
  process.stderr.write("usage: site_reference_oracle.js <SkudPaiSho checkout>\n");
  process.exit(2);
}

const actualCommit = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: referenceRoot,
  encoding: "utf8",
}).trim();
if (actualCommit !== EXPECTED_COMMIT) {
  throw new Error(`expected source commit ${EXPECTED_COMMIT}, got ${actualCommit}`);
}

globalThis.window = {
  location: {
    href: "https://skudpaisho.com/",
    replace() {},
  },
  navigator: {
    userAgent: "Node.js differential oracle",
    vendor: "",
  },
};
globalThis.QueryString = {};
globalThis.BRAND_NEW = "Brand New";
globalThis.WAITING_FOR_ENDPOINT = "Waiting for endpoint";
globalThis.showBadMoveModal = () => {};

function load(relativePath) {
  const filename = path.join(referenceRoot, relativePath);
  vm.runInThisContext(fs.readFileSync(filename, "utf8"), { filename });
}

[
  "js/GameData.js",
  "js/CommonNotationObjects.js",
  "js/GameOptions.js",
  "js/pai-sho-common/PaiShoBoardHelp.js",
  "js/pai-sho-common/PaiShoMarkingManager.js",
  "js/skud-pai-sho/SkudPaiShoBoardPoint.js",
  "js/skud-pai-sho/SkudPaiShoTile.js",
  "js/skud-pai-sho/SkudPaiShoHarmony.js",
  "js/skud-pai-sho/SkudPaiShoBoard.js",
  "js/skud-pai-sho/SkudPaiShoTileManager.js",
  "js/skud-pai-sho/SkudPaiShoGameNotation.js",
  "js/skud-pai-sho/SkudPaiShoGameManager.js",
  "js/ai/SkudAIv1.js",
].forEach(load);

const sourceGetMoveScore = SkudAIv1.prototype.getMoveScore;
SkudAIv1.prototype.getMoveScore = function (game, move, depth) {
  try {
    return sourceGetMoveScore.call(this, game, move, depth);
  } catch (error) {
    throw new Error(`source failed while scoring ${move.fullMoveText}`, {
      cause: error,
    });
  }
};

globalThis.ggOptions = [];
globalThis.getPlayerCodeFromName = (player) => (player === HOST ? "H" : "G");
globalThis.getOpponentName = (player) => (player === HOST ? GUEST : HOST);

function newStandardGame(startingFlower = "R3") {
  const game = new SkudPaiShoGameManager(null, true, true);
  [
    "0H.R,W,K,B",
    "0G.R,W,K,B",
    `1G.${startingFlower}(0,-8)`,
    `1H.${startingFlower}(0,8)`,
  ].forEach((text) => game.runNotationMove(new SkudPaiShoNotationMove(text)));
  return game;
}

function pointAt(game, notation) {
  const point = new NotationPoint(notation).rowAndColumn;
  return game.board.cells[point.row][point.col];
}

function replayAndRequireBonus(priorMoves) {
  const game = newStandardGame();
  let bonusAllowed = false;
  priorMoves.forEach((text, index) => {
    bonusAllowed = game.runNotationMove(new SkudPaiShoNotationMove(text));
    if (index === priorMoves.length - 1 && !bonusAllowed) {
      throw new Error(`expected Harmony Bonus for ${text}`);
    }
  });
  return game;
}

function requireTile(game, notation, code) {
  const point = pointAt(game, notation);
  if (!point.hasTile() || point.tile.code !== code) {
    throw new Error(`expected ${code} at ${notation}`);
  }
  return point.tile;
}

function actionText(move) {
  if (move.moveType === PLANTING) {
    return `plant ${move.plantedFlowerType} ${move.endPoint.pointText}`;
  }
  if (move.moveType === ARRANGING) {
    return `arrange ${move.startPoint.pointText} ${move.endPoint.pointText}`;
  }
  throw new Error(`unsupported oracle move ${move.fullMoveText}`);
}

function decisionTexts(game, move) {
  const normalized = new SkudPaiShoNotationMove(move.fullMoveText);
  const decisions = [actionText(normalized)];
  const after = game.getCopy();
  const bonusAllowed = after.runNotationMove(normalized);
  if (!bonusAllowed) {
    return decisions;
  }
  if (!normalized.hasHarmonyBonus()) {
    if (
      after.board.winners.length === 0 &&
      after.endGameWinners.length === 0
    ) {
      decisions.push("skip-bonus");
    }
    return decisions;
  }

  const code = normalized.bonusTileCode;
  const point = normalized.bonusEndPoint.pointText;
  if (/^[RW][345]$/.test(code)) {
    decisions.push(`bonus-plant ${code} ${point}`);
  } else if (code === "L" || code === "O") {
    decisions.push(`plant-special ${code} ${point}`);
  } else if (code === "B" && normalized.boatBonusPoint) {
    decisions.push(`accent B move ${point} ${normalized.boatBonusPoint.pointText}`);
  } else {
    decisions.push(`accent ${code} at ${point}`);
  }
  return decisions;
}

function characterize(name, priorMoves, player, moveNumber, startingFlower = "R3") {
  const game = newStandardGame(startingFlower);
  priorMoves.forEach((text) => game.runNotationMove(new SkudPaiShoNotationMove(text)));
  const bot = new SkudAIv1();
  bot.setPlayer(player);
  bot.moveNum = moveNumber;
  const moves = bot.getPossibleMoves(game, player);
  const actions = moves.map((move) => ({
    action: actionText(move),
    score: bot.getMoveScore(game, move, bot.scoreDepth).get(bot.scoreDepth),
  }));
  return { name, player, moveNumber, actions };
}

function splitMix64(seedText) {
  const mask = (1n << 64n) - 1n;
  let state = BigInt(seedText);
  return () => {
    state = (state + 0x9e3779b97f4a7c15n) & mask;
    let value = state;
    value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
    value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask;
    value ^= value >> 31n;
    return Number(value >> 11n) / 9007199254740992;
  };
}

function selections(name, priorMoves, player, moveNumber, startingFlower = "R3") {
  return ["0x0", "0x1", "0x3", "0x2a", "0x4755455354"].map((seed) => {
    const game = newStandardGame(startingFlower);
    priorMoves.forEach((text) => game.runNotationMove(new SkudPaiShoNotationMove(text)));
    const bot = new SkudAIv1();
    bot.setPlayer(player);
    Math.random = splitMix64(seed);
    const move = bot.getMove(game.getCopy(), moveNumber);
    return { scenario: name, seed, decisions: decisionTexts(game, move) };
  });
}

function seededTrajectory({
  name,
  startingFlower,
  hostSeed,
  guestSeed,
  decisionSoftLimit,
  debugMove = null,
}) {
  const game = newStandardGame(startingFlower);
  const bots = {
    [HOST]: new SkudAIv1(),
    [GUEST]: new SkudAIv1(),
  };
  bots[HOST].setPlayer(HOST);
  bots[GUEST].setPlayer(GUEST);
  const randomByPlayer = {
    [HOST]: splitMix64(hostSeed),
    [GUEST]: splitMix64(guestSeed),
  };
  const originalRandom = Math.random;
  const decisions = [];
  const turns = [];
  let moveDebug;
  let player = GUEST;
  let moveNumber = 2;

  try {
    while (
      decisions.length < decisionSoftLimit &&
      game.board.winners.length === 0 &&
      game.endGameWinners.length === 0
    ) {
      const probe = new SkudAIv1();
      probe.setPlayer(player);
      probe.moveNum = moveNumber;
      const possibleActions = probe
        .getPossibleMoves(game.getCopy(), player)
        .map(actionText);
      if (debugMove && decisions.length === debugMove.decisionOffset) {
        moveDebug = explainArrangement(
          game,
          player,
          debugMove.from,
          debugMove.to,
        );
      }
      Math.random = randomByPlayer[player];
      let move;
      try {
        // The browser controller gives the bot a disposable copy. This is
        // observable because ensurePlant leaves POSSIBLE_MOVE markers behind.
        move = bots[player].getMove(game.getCopy(), moveNumber);
      } catch (error) {
        throw new Error(
          `${name} failed for ${player} move ${moveNumber} after ${decisions.length} decisions; recent decisions: ${decisions.slice(-6).join(" | ")}`,
          { cause: error },
        );
      }
      if (!move) {
        throw new Error(
          `${name} returned no move for ${player} after ${decisions.length} decisions; a partial trajectory cannot be emitted as verified`,
        );
      }
      const normalized = new SkudPaiShoNotationMove(move.fullMoveText);
      const selectedAction = actionText(normalized);
      turns.push({
        decisionOffset: decisions.length,
        player: player === HOST ? "H" : "G",
        moveNumber,
        actionCount: possibleActions.length,
        actionDigest: sha256Lines(possibleActions),
        selectedIndex: possibleActions.indexOf(selectedAction),
        possibleActions,
      });
      decisions.push(...decisionTexts(game, normalized));
      game.runNotationMove(normalized);
      if (player === GUEST) {
        player = HOST;
      } else {
        player = GUEST;
        moveNumber += 1;
      }
    }
  } finally {
    Math.random = originalRandom;
  }

  let termination = "DECISION_LIMIT";
  if (game.board.winners.length > 0) {
    termination = `RING_${winnerCode(game.board.winners)}`;
  } else if (game.endGameWinners.length > 0) {
    termination = `RESERVE_${winnerCode(game.endGameWinners)}`;
  }
  return {
    name,
    startingFlower,
    hostSeed,
    guestSeed,
    decisionSoftLimit,
    termination,
    turns,
    decisions,
    moveDebug,
  };
}

function explainArrangement(game, player, fromNotation, toNotation) {
  const from = pointAt(game, fromNotation);
  const to = pointAt(game, toNotation);
  const tile = from.tile;
  const canCapture = to.hasTile() && game.board.canCapture(from, to);
  return {
    player,
    from: fromNotation,
    to: toNotation,
    tile: tile && `${tile.ownerName}:${tile.code}`,
    trapped: tile && tile.trapped,
    drained: tile && tile.drained,
    destinationTypes: to.types,
    destinationTile: to.hasTile() && `${to.tile.ownerName}:${to.tile.code}`,
    canCapture,
    canHold: tile && to.canHoldTile(tile, canCapture),
    distance: Math.abs(from.row - to.row) + Math.abs(from.col - to.col),
    movement: tile && tile.getMoveDistance(),
    reachable: tile && game.board.verifyAbleToReach(from, to, tile.getMoveDistance()),
    createsDisharmony: tile && game.board.moveCreatesDisharmony(from, to),
    accepted: game.board.canMoveTileToPoint(player, from, to),
  };
}

function sha256Lines(lines) {
  return crypto.createHash("sha256").update(lines.join("\n")).digest("hex");
}

function winnerCode(winners) {
  if (winners.length !== 1) {
    return "DRAW";
  }
  return winners[0] === HOST ? "HOST" : "GUEST";
}

const TRAJECTORY_SPECS = [
  {
    name: "red3-reference",
    startingFlower: "R3",
    hostSeed: "0x484f5354",
    guestSeed: "0x4755455354",
    decisionSoftLimit: 512,
  },
  {
    name: "white4-independent",
    startingFlower: "W4",
    hostSeed: "0x1",
    guestSeed: "0x2a",
    decisionSoftLimit: 512,
  },
];

function trajectoryFixture() {
  return TRAJECTORY_SPECS.map(seededTrajectory);
}

function reserveExhaustionContract() {
  const game = newStandardGame();
  const lastBasic = game.tileManager.guestTiles.find(
    (tile) => tile.type === BASIC_FLOWER && tile.code === "W4",
  );
  game.tileManager.guestTiles = game.tileManager.guestTiles.filter(
    (tile) => tile.type !== BASIC_FLOWER,
  );
  game.tileManager.guestTiles.push(lastBasic);

  const bot = new SkudAIv1();
  bot.setPlayer(GUEST);
  bot.moveNum = 2;
  const move = bot
    .getPossibleMoves(game, GUEST)
    .find((candidate) => candidate.moveType === PLANTING);
  const after = game.getCopy();
  after.runNotationMove(move);
  return {
    action: actionText(move),
    score: bot.calculateScore(game, after),
    boardWinnerCount: after.board.winners.length,
    endGameWinnerCount: after.endGameWinners.length,
  };
}

function terminalBonusContract() {
  const game = newStandardGame();
  [
    "2G.(0,-8)-(0,-5)",
    "2H.(0,8)-(0,5)",
    "3G.R4(0,-8)",
    "3H.R4(0,8)",
    "4G.(0,-8)-(-1,-5)",
    "4H.(0,8)-(1,5)",
    "5G.(-1,-5)-(-1,-4)",
    "5H.(1,5)-(1,4)",
  ].forEach((text) => game.runNotationMove(new SkudPaiShoNotationMove(text)));
  const lastBasic = game.tileManager.guestTiles.find(
    (tile) => tile.type === BASIC_FLOWER && tile.code === "W5",
  );
  if (!lastBasic) {
    throw new Error("terminal bonus fixture requires one Guest W5");
  }
  game.tileManager.guestTiles = game.tileManager.guestTiles.filter(
    (tile) => tile.type !== BASIC_FLOWER,
  );
  game.tileManager.guestTiles.push(lastBasic);

  const move = new SkudPaiShoNotationMove(
    "6G.(-1,-4)-(-1,-5)+W5(0,-8)",
  );
  const decisions = decisionTexts(game, move);
  const bonusAllowed = game.runNotationMove(move);
  const remainingBasics = game.tileManager.guestTiles.filter(
    (tile) => tile.type === BASIC_FLOWER,
  ).length;
  requireTile(game, "0,-8", "W5");
  if (
    !bonusAllowed ||
    decisions.join("|") !==
      "arrange -1,-4 -1,-5|bonus-plant W5 0,-8" ||
    remainingBasics !== 0 ||
    game.board.winners.length !== 0 ||
    game.endGameWinners.length !== 2
  ) {
    throw new Error("terminal Basic bonus contract changed");
  }
  return {
    decisions,
    bonusAllowed,
    remainingBasics,
    boardWinnerCount: game.board.winners.length,
    endGameWinnerCount: game.endGameWinners.length,
  };
}

function mixedCycleContract() {
  const game = new SkudPaiShoGameManager(null, true, true);
  [
    [HOST, "L", "-4,-2"],
    [HOST, "R3", "-4,2"],
    [HOST, "R4", "0,3"],
    [HOST, "O", "2,-3"],
    [GUEST, "L", "4,2"],
    [GUEST, "R3", "4,-2"],
    [GUEST, "R4", "0,-2"],
  ].forEach(([player, code, notation]) => {
    const tile = game.tileManager.grabTile(player, code);
    game.board.placeTile(tile, new NotationPoint(notation), game.tileManager);
  });

  const bot = new SkudAIv1();
  bot.setPlayer(HOST);
  const beforeCycle = game.board.harmonyManager.ringLengthForPlayer(HOST);
  const beforeSurroundness = game.board.getSurroundness(HOST);
  const move = new SkudPaiShoNotationMove("2H.(0,3)-(0,2)");
  const after = game.getCopy();
  after.runNotationMove(move);
  return {
    action: actionText(move),
    beforeCycle,
    afterCycle: after.board.harmonyManager.ringLengthForPlayer(HOST),
    beforeSurroundness,
    afterSurroundness: after.board.getSurroundness(HOST),
    score: bot.calculateScore(game, after),
    winnerCount: after.board.winners.length,
  };
}

const developedMoves = [
  "2G.(0,-8)-(0,-5)",
  "2H.(0,8)-(0,5)",
  "3G.R4(0,-8)",
  "3H.R4(0,8)",
  "4G.(0,-8)-(-1,-5)",
  "4H.(0,8)-(1,5)",
  "5G.(-1,-5)-(-1,-4)",
  "5H.(1,5)-(1,4)",
];

const developedRockMoves = [
  ...developedMoves,
  "6G.(-1,-4)-(-1,-5)+R(0,0)",
];

const developedKnotweedMoves = [
  ...developedMoves,
  "6G.(-1,-4)-(-1,-5)+K(-1,-4)",
];

const developedWheelMoves = [
  ...developedMoves,
  "6G.(-1,-4)-(-1,-5)+W(-1,-4)",
];

const developedBoatMoves = [
  ...developedMoves,
  "6G.(-1,-4)-(-1,-5)+B(0,-5)-(0,-4)",
];

const rockGame = replayAndRequireBonus(developedRockMoves);
requireTile(rockGame, "0,0", "R");

const knotweedGame = replayAndRequireBonus(developedKnotweedMoves);
requireTile(knotweedGame, "-1,-4", "K");
if (!requireTile(knotweedGame, "-1,-5", "R4").drained) {
  throw new Error("expected Knotweed to drain R4 at -1,-5");
}
if (!requireTile(knotweedGame, "0,-5", "R3").drained) {
  throw new Error("expected Knotweed to drain R3 at 0,-5");
}

const wheelGame = replayAndRequireBonus(developedWheelMoves);
requireTile(wheelGame, "-1,-4", "W");
requireTile(wheelGame, "-1,-5", "R3");
requireTile(wheelGame, "-2,-5", "R4");
if (pointAt(wheelGame, "0,-5").hasTile()) {
  throw new Error("expected Wheel to vacate 0,-5");
}

const boatGame = replayAndRequireBonus(developedBoatMoves);
requireTile(boatGame, "0,-5", "B");
requireTile(boatGame, "0,-4", "R3");

const captureReadyMoves = [
  "2G.W3(-8,0)",
  "2H.(0,8)-(-2,7)",
  "3G.(-8,0)-(-6,1)",
  "3H.(-2,7)-(-4,6)",
  "4G.(-6,1)-(-5,3)",
  "4H.(-4,6)-(-4,4)",
];

const ringReadyMoves = [
  "2G.(0,-8)-(-2,-6)",
  "2H.(0,8)-(1,8)",
  "3G.R5(0,-8)",
  "3H.R4(-8,0)",
  "4G.(0,-8)-(1,-6)+W3(0,8)",
  "4H.(-8,0)-(-7,1)",
  "5G.(0,8)-(1,6)+R5(0,8)",
  "5H.W4(0,-8)",
];

const ringGame = newStandardGame("R4");
ringReadyMoves.forEach((text) => ringGame.runNotationMove(new SkudPaiShoNotationMove(text)));
const ringBot = new SkudAIv1();
ringBot.setPlayer(GUEST);
ringBot.moveNum = 6;
const ringMove = ringBot
  .getPossibleMoves(ringGame, GUEST)
  .find((move) => actionText(move) === "arrange 0,8 -2,6");
if (!ringMove) {
  throw new Error("expected the archived ring-closing move to be legal on the source engine");
}
const ringAfter = ringGame.getCopy();
ringAfter.runNotationMove(ringMove);
if (!ringAfter.board.winners.includes(GUEST)) {
  throw new Error("expected the archived move to create a Guest Harmony Ring");
}
if (ringBot.calculateScore(ringGame, ringAfter) !== 9_999_999) {
  throw new Error("expected the site bot's Harmony Ring score");
}

const scenarios = [
  characterize("formal-opening", [], GUEST, 2),
  characterize("developed-no-bonus", developedMoves, GUEST, 6),
  characterize("developed-rock", developedRockMoves, HOST, 6),
  characterize("developed-knotweed", developedKnotweedMoves, HOST, 6),
  characterize("developed-wheel", developedWheelMoves, HOST, 6),
  characterize("developed-boat", developedBoatMoves, HOST, 6),
  characterize("capture-ready", captureReadyMoves, GUEST, 5),
  characterize("ring-ready", ringReadyMoves, GUEST, 6, "R4"),
];
const selectedMoves = [
  ...selections("formal-opening", [], GUEST, 2),
  ...selections("developed-no-bonus", developedMoves, GUEST, 6),
  ...selections("developed-rock", developedRockMoves, HOST, 6),
  ...selections("developed-knotweed", developedKnotweedMoves, HOST, 6),
  ...selections("developed-wheel", developedWheelMoves, HOST, 6),
  ...selections("developed-boat", developedBoatMoves, HOST, 6),
  ...selections("capture-ready", captureReadyMoves, GUEST, 5),
  ...selections("ring-ready", ringReadyMoves, GUEST, 6, "R4"),
];
const reserveFinish = reserveExhaustionContract();
const terminalBonus = terminalBonusContract();
const mixedCycle = mixedCycleContract();

if (process.argv[3] === "--move-debug") {
  const trajectoryName = process.argv[4];
  const spec = TRAJECTORY_SPECS.find(
    (candidate) => candidate.name === trajectoryName,
  );
  if (!spec) {
    throw new Error(`unknown trajectory ${trajectoryName}`);
  }
  const trajectory = seededTrajectory({
    ...spec,
    debugMove: {
      decisionOffset: Number.parseInt(process.argv[5], 10),
      from: process.argv[6],
      to: process.argv[7],
    },
  });
  process.stdout.write(JSON.stringify(trajectory.moveDebug, null, 2) + "\n");
} else if (process.argv[3] === "--trajectory-debug") {
  const trajectoryName = process.argv[4];
  const decisionOffset = Number.parseInt(process.argv[5], 10);
  const trajectory = trajectoryFixture().find(
    (candidate) => candidate.name === trajectoryName,
  );
  if (!trajectory) {
    throw new Error(`unknown trajectory ${trajectoryName}`);
  }
  const turn = trajectory.turns.find(
    (candidate) => candidate.decisionOffset === decisionOffset,
  );
  if (!turn) {
    throw new Error(`no turn at decision offset ${decisionOffset}`);
  }
  process.stdout.write(turn.possibleActions.join("\n") + "\n");
} else if (process.argv[3] === "--trajectory-fixture") {
  process.stdout.write("PAISHO-SITE-TRAJECTORIES\t1\n");
  process.stdout.write(`# source-commit ${EXPECTED_COMMIT}\n`);
  trajectoryFixture().forEach((trajectory) => {
    process.stdout.write(`[trajectory ${trajectory.name}]\n`);
    process.stdout.write(`starting_flower\t${trajectory.startingFlower}\n`);
    process.stdout.write(`host_seed\t${trajectory.hostSeed}\n`);
    process.stdout.write(`guest_seed\t${trajectory.guestSeed}\n`);
    process.stdout.write(`decision_soft_limit\t${trajectory.decisionSoftLimit}\n`);
    process.stdout.write(`termination\t${trajectory.termination}\n`);
    trajectory.turns.forEach((turn) => {
      process.stdout.write(
        `turn\t${turn.decisionOffset}\t${turn.player}\t${turn.moveNumber}\t${turn.actionCount}\t${turn.actionDigest}\t${turn.selectedIndex}\n`,
      );
    });
    trajectory.decisions.forEach((decision) => {
      process.stdout.write(`action\t${decision}\n`);
    });
  });
} else if (process.argv[3] === "--fixture") {
  process.stdout.write(`# source-commit ${EXPECTED_COMMIT}\n`);
  scenarios.forEach((scenario) => {
    process.stdout.write(`[${scenario.name}]\n`);
    scenario.actions.forEach(({ action, score }) => {
      process.stdout.write(`${action}\t${score}\n`);
    });
  });
  process.stdout.write("[seeded-selections]\n");
  selectedMoves.forEach(({ scenario, seed, decisions }) => {
    process.stdout.write(`${scenario}\t${seed}\t${decisions.join("\t")}\n`);
  });
  process.stdout.write("[reserve-exhaustion-contract]\n");
  process.stdout.write(
    `${reserveFinish.action}\t${reserveFinish.score}\t${reserveFinish.boardWinnerCount}\t${reserveFinish.endGameWinnerCount}\n`,
  );
  process.stdout.write("[terminal-bonus-contract]\n");
  process.stdout.write(
    `${terminalBonus.decisions.join("\t")}\t${Number(terminalBonus.bonusAllowed)}\t${terminalBonus.remainingBasics}\t${terminalBonus.boardWinnerCount}\t${terminalBonus.endGameWinnerCount}\n`,
  );
  process.stdout.write("[mixed-cycle-contract]\n");
  process.stdout.write(
    `${mixedCycle.action}\t${mixedCycle.beforeCycle}\t${mixedCycle.afterCycle}\t${mixedCycle.beforeSurroundness}\t${mixedCycle.afterSurroundness}\t${mixedCycle.score}\t${mixedCycle.winnerCount}\n`,
  );
} else {
  process.stdout.write(
    `${JSON.stringify({ sourceCommit: EXPECTED_COMMIT, scenarios, selectedMoves, reserveFinish, terminalBonus, mixedCycle }, null, 2)}\n`,
  );
}
