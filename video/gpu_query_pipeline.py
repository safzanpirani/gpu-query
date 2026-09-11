"""gpu-query explainer film.

Every activation, score and label on screen is read from `trace.json`, exported
from a trained checkpoint by `export_trace.py`. Nothing is invented. Where
channels or rows are sampled for legibility the bottom caption says so.

Look mixes the two reference films: gpu-time's white ground, Geist Mono, scene
counter and honesty caption; gpu-lexer's persistent token strip and activation
heatmaps instead of fully-connected edges.

Render:
  manim-env/bin/manim -qh gpu_query_pipeline.py GpuQueryPipeline
"""

from __future__ import annotations

import json
from pathlib import Path

from manim import *
from manim_voiceover import VoiceoverScene

from gemini_voice import GeminiTTSService

HERE = Path(__file__).resolve().parent
TRACE = json.loads((HERE / "trace.json").read_text())

PAPER = "#FCFCFD"
INK = "#16161D"
MUTED = "#9A9AA8"
RULE = "#E3E3EA"
BLUE = "#2563EB"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
AMBER = "#B45309"

MONO = "Geist Mono"
SCENES = 8

config.background_color = PAPER

DEMO = TRACE["demo"]
ACT = TRACE["activations"]
CKPT = TRACE["checkpoint"]
TOKENS = DEMO["tokens"]
PRED = DEMO["predictedLabels"]
RESOLVED = DEMO["resolved"]

ROLE_COLOR = {
    "O": MUTED,
    "FIELD": BLUE,
    "VALUE": VIOLET,
    "VALUE_CONT": VIOLET,
    "OP_EQ": TEAL,
    "OP_LT": TEAL,
    "OP_GT": TEAL,
    "OP_CONTAINS": TEAL,
    "NEG": AMBER,
    "AND": INK,
    "OR": INK,
}


def mono(text, size=26, color=INK, weight=NORMAL):
    return Text(text, font=MONO, font_size=size, color=color, weight=weight)


def token_box(label, width=None, color=INK, fill=0.0):
    txt = mono(label, size=24, color=INK)
    w = (width or txt.width + 0.42)
    rect = RoundedRectangle(
        corner_radius=0.09, width=w, height=0.62,
        stroke_color=color, stroke_width=1.6,
        fill_color=color, fill_opacity=fill,
    )
    txt.move_to(rect)
    return VGroup(rect, txt)


def token_strip(tokens, scale=1.0):
    boxes = VGroup(*[token_box(t) for t in tokens]).arrange(RIGHT, buff=0.14)
    if boxes.width > 12.6:
        boxes.scale_to_fit_width(12.6)
    return boxes.scale(scale)


def heat(value, lo=-1.0, hi=1.0):
    """Signed activation to a colour. Violet negative, blue positive."""
    span = max(1e-6, hi - lo)
    t = min(1.0, max(0.0, (value - lo) / span))
    if t >= 0.5:
        return interpolate_color(ManimColor(RULE), ManimColor(BLUE), (t - 0.5) * 2)
    return interpolate_color(ManimColor(VIOLET), ManimColor(RULE), t * 2)


def channel_column(values, rows=8, cell=0.13, gap=0.02):
    """A sampled activation column. Caller must say it is sampled."""
    picks = values[:rows]
    squares = VGroup()
    for v in picks:
        s = Square(side_length=cell, stroke_width=0, fill_opacity=1.0,
                   fill_color=heat(v))
        squares.add(s)
    squares.arrange(DOWN, buff=gap)
    return squares


class GpuQueryPipeline(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            GeminiTTSService(voice="Charon", cache_dir="tts-cache")
        )
        self.camera.background_color = PAPER
        self.index = 0
        self.chrome = VGroup()
        self.add_chrome()

        self.scene_ask()
        self.scene_scan()
        self.scene_schema()
        self.scene_context()
        self.scene_scores()
        self.scene_compile()
        self.scene_result()
        self.scene_lockup()

    # ---------- chrome ----------

    def add_chrome(self):
        mark = mono("gpu-query", size=17, color=MUTED).to_corner(UL, buff=0.42)
        self.counter = mono(f"01 / 0{SCENES}", size=17, color=MUTED).to_corner(
            UR, buff=0.42
        )
        self.chrome = VGroup(mark, self.counter)
        self.add(self.chrome)

    def begin(self, number, title, caption=None):
        """Advance the counter, set the title and the honesty caption."""
        self.index = number
        new_counter = mono(f"0{number} / 0{SCENES}", size=17, color=MUTED).to_corner(
            UR, buff=0.42
        )
        head = mono(title, size=30, color=INK).to_edge(UP, buff=0.95)
        parts = [Transform(self.counter, new_counter), FadeIn(head, shift=DOWN * 0.15)]
        foot = None
        if caption:
            foot = mono(caption, size=16, color=MUTED)
            if foot.width > 12.4:
                foot.scale_to_fit_width(12.4)
            foot.to_edge(DOWN, buff=0.42)
            parts.append(FadeIn(foot))
        self.play(*parts, run_time=0.5)
        return head, foot

    def clear_stage(self, *keep):
        junk = [
            m for m in self.mobjects
            if m not in self.chrome and m is not self.counter and m not in keep
        ]
        if junk:
            self.play(*[FadeOut(m) for m in junk], run_time=0.4)

    # ---------- scenes ----------

    def scene_ask(self):
        head, _ = self.begin(1, "A search bar")
        bar = RoundedRectangle(
            corner_radius=0.14, width=10.4, height=1.15,
            stroke_color=RULE, stroke_width=2.2,
            fill_color=WHITE, fill_opacity=1.0,
        ).shift(UP * 0.25)
        typed = mono(DEMO["text"], size=25, color=INK)
        if typed.width > 9.6:
            typed.scale_to_fit_width(9.6)
        typed.move_to(bar)

        paths = VGroup(
            mono("remember the query syntax", size=20, color=MUTED),
            mono("or send it to a language model", size=20, color=MUTED),
        ).arrange(DOWN, buff=0.3).next_to(bar, DOWN, buff=0.85)
        cross = VGroup(
            Line(paths[0].get_left(), paths[0].get_right(), color=AMBER, stroke_width=2),
            Line(paths[1].get_left(), paths[1].get_right(), color=AMBER, stroke_width=2),
        )
        verdict = mono("no request leaves the page", size=22, color=TEAL).move_to(
            paths.get_center()
        )

        with self.voiceover(
            "Your app has a search bar. Today it needs either syntax your users "
            "have to memorize, or a round trip to a language model. Neither is "
            "necessary."
        ):
            self.play(Create(bar), run_time=0.7)
            self.play(AddTextLetterByLetter(typed, run_time=1.7))
            self.play(LaggedStart(*[FadeIn(p) for p in paths], lag_ratio=0.35),
                      run_time=1.1)
            self.play(Create(cross), run_time=0.6)
            self.play(FadeOut(paths), FadeOut(cross), FadeIn(verdict), run_time=0.7)

        self.strip = token_strip(TOKENS).move_to(ORIGIN)
        self.play(
            FadeOut(bar), FadeOut(verdict), FadeOut(head),
            ReplacementTransform(typed, self.strip), run_time=0.9,
        )

    def scene_scan(self):
        head, foot = self.begin(
            2, "One mechanical scan",
            "Real feature rows. Row blocks shown for three tokens; each token has up to 16.",
        )
        self.play(self.strip.animate.shift(UP * 0.55), run_time=0.5)

        picks = [1, 4, 6]
        pitch = self.strip[1].get_center()[0] - self.strip[0].get_center()[0]
        cards = VGroup()
        for i in picks:
            rows = DEMO["featureRows"][i][:5]
            lines = VGroup(*[
                mono(f"{r['block']}+{r['offset']}", size=13, color=MUTED)
                for r in rows
            ]).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            # Keep each card inside its own token pitch so they cannot collide.
            if lines.width > abs(pitch) * 2.4:
                lines.scale_to_fit_width(abs(pitch) * 2.4)
            lines.next_to(self.strip[i], DOWN, buff=0.55)
            cards.add(lines)

        with self.voiceover(
            "One mechanical pass splits the phrase into tokens, and emits a "
            "handful of sparse feature rows for each one. Shape, length, case. "
            "No grammar, no regular expressions."
        ):
            self.play(LaggedStart(*[
                self.strip[i].animate.set_stroke(BLUE, 2.2) for i in picks
            ], lag_ratio=0.2), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cards],
                                  lag_ratio=0.25), run_time=1.6)
            self.wait(0.5)

        self.play(FadeOut(cards), FadeOut(head), FadeOut(foot),
                  *[self.strip[i].animate.set_stroke(INK, 1.6) for i in picks],
                  run_time=0.5)

    def scene_schema(self):
        head, foot = self.begin(
            3, "The schema stays outside",
            "Real match rows from the featurizer. Field names never enter the weights.",
        )
        self.play(self.strip.animate.scale(0.82).to_edge(DOWN, buff=1.15), run_time=0.5)

        fields = DEMO["schema"][:4]
        rows = VGroup(*[
            mono(f"{f['name']}: {f['kind']}", size=19, color=INK) for f in fields
        ]).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        panel = RoundedRectangle(
            corner_radius=0.1, width=rows.width + 0.7, height=rows.height + 0.7,
            stroke_color=RULE, stroke_width=1.8, fill_color=WHITE, fill_opacity=1.0,
        )
        rows.move_to(panel)
        schema_panel = VGroup(panel, rows).to_edge(LEFT, buff=1.0).shift(UP * 0.35)
        schema_label = mono("your schema", size=16, color=MUTED).next_to(
            schema_panel, UP, buff=0.2
        )

        wall = DashedLine(
            UP * 1.9, DOWN * 1.5, color=AMBER, stroke_width=2.4, dash_length=0.12
        ).shift(RIGHT * 0.35)
        wall_label = mono("model boundary", size=15, color=AMBER).next_to(
            wall, UP, buff=0.15
        )

        arrows = VGroup(*[
            Arrow(schema_panel.get_right() + UP * (0.55 - 0.45 * i),
                  wall.get_center() + UP * (0.75 - 0.45 * i),
                  buff=0.12, stroke_width=2.2, color=MUTED,
                  max_tip_length_to_length_ratio=0.14)
            for i in range(3)
        ])

        passed = VGroup(
            mono("matched a field  ·  kind text", size=18, color=BLUE),
            mono("matched value of nearest field", size=18, color=VIOLET),
            mono("distance to field: 1", size=18, color=TEAL),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT).next_to(wall, RIGHT, buff=0.6)

        punch = mono("no field name crosses", size=20, color=AMBER).next_to(
            passed, DOWN, buff=0.5
        )

        with self.voiceover(
            "Here is the part that matters. The schema reaches the featurizer, "
            "but the field names are stripped at the boundary. The model only "
            "learns that a token matched some field, of some kind. It never sees "
            "the word itself."
        ):
            self.play(FadeIn(schema_panel), FadeIn(schema_label), run_time=0.8)
            self.play(Create(wall), FadeIn(wall_label), run_time=0.6)
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                      run_time=1.0)
            self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in passed],
                                  lag_ratio=0.3), run_time=1.6)
            self.play(FadeIn(punch), run_time=0.6)
            self.wait(0.8)

        self.clear_stage(self.strip)

    def scene_context(self):
        head, foot = self.begin(
            4, "Context, both directions",
            "Real gated states. Eight of thirty-two channels shown per token.",
        )
        self.play(self.strip.animate.scale(1 / 0.82).move_to(DOWN * 1.45), run_time=0.5)

        forward = ACT["forward"]
        backward = ACT["backward"]
        columns = VGroup()
        for i in range(len(TOKENS)):
            col = channel_column(forward[i])
            col.next_to(self.strip[i], UP, buff=0.45)
            columns.add(col)

        formula = mono("state[t] = gate · state[t-1] + candidate", size=21, color=INK)
        formula.next_to(columns, UP, buff=0.55)

        sweep = Rectangle(
            width=0.72, height=1.5, stroke_color=BLUE, stroke_width=2.2,
            fill_color=BLUE, fill_opacity=0.07,
        ).move_to(columns[0])

        with self.voiceover(
            "Gated state flows forward across the phrase, then backward. By the "
            "end every token knows what surrounds it on both sides. These are the "
            "actual state values."
        ):
            self.play(LaggedStart(*[FadeIn(c) for c in columns], lag_ratio=0.1),
                      run_time=1.2)
            self.play(FadeIn(formula), run_time=0.5)
            self.play(Create(sweep), run_time=0.3)
            self.play(sweep.animate.move_to(columns[-1]), run_time=1.5,
                      rate_func=rate_functions.ease_in_out_sine)
            back_cols = VGroup(*[
                channel_column(backward[i]).move_to(columns[i])
                for i in range(len(TOKENS))
            ])
            self.play(sweep.animate.set_stroke(VIOLET).set_fill(VIOLET, 0.07),
                      run_time=0.3)
            self.play(
                sweep.animate.move_to(columns[0]),
                Transform(columns, back_cols),
                run_time=1.6, rate_func=rate_functions.ease_in_out_sine,
            )
            self.play(FadeOut(sweep), run_time=0.3)

        self.play(FadeOut(columns), FadeOut(formula), FadeOut(head), FadeOut(foot),
                  run_time=0.5)

    def scene_scores(self):
        target = TOKENS.index("listens")
        head, foot = self.begin(
            5, "Lookup says field. Context says value.",
            f"Real logits for the token '{TOKENS[target]}'. All twelve roles shown.",
        )
        self.play(self.strip.animate.to_edge(DOWN, buff=0.95), run_time=0.5)

        note = VGroup(
            mono("'listens' exactly matches the 'plays' field", size=19, color=MUTED),
            mono("but here it is a value", size=19, color=VIOLET),
        ).arrange(DOWN, buff=0.18).next_to(self.strip, UP, buff=0.45)

        logits = ACT["logits"][target]
        labels = ACT["labels"]
        order = sorted(range(len(labels)), key=lambda i: -logits[i])[:6]
        lo = min(logits)
        span = max(1e-6, max(logits) - lo)

        bars = VGroup()
        for rank, idx in enumerate(order):
            width = 0.25 + 5.4 * ((logits[idx] - lo) / span)
            colour = ROLE_COLOR.get(labels[idx], MUTED)
            bar = Rectangle(width=width, height=0.3, stroke_width=0,
                            fill_color=colour, fill_opacity=0.85)
            name = mono(labels[idx], size=17, color=INK)
            value = mono(f"{logits[idx]:+.2f}", size=15, color=MUTED)
            name.next_to(bar, LEFT, buff=0.25)
            value.next_to(bar, RIGHT, buff=0.2)
            row = VGroup(name, bar, value)
            bars.add(row)
        bars.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        for row in bars:
            row[0].next_to(row[1], LEFT, buff=0.25)
            row[2].next_to(row[1], RIGHT, buff=0.2)
        bars.move_to(UP * 0.75)

        with self.voiceover(
            "For the word listens, a lookup says field, because it matches a "
            "field name exactly. Context says value. These are the actual scores, "
            "and value wins."
        ):
            self.play(self.strip[target].animate.set_stroke(VIOLET, 2.6), run_time=0.4)
            self.play(FadeIn(note), run_time=0.8)
            self.play(LaggedStart(*[
                GrowFromEdge(row[1], LEFT) for row in bars
            ], lag_ratio=0.16), run_time=1.6)
            self.play(LaggedStart(*[
                FadeIn(VGroup(row[0], row[2])) for row in bars
            ], lag_ratio=0.1), run_time=0.8)
            box = SurroundingRectangle(bars[0], color=VIOLET, buff=0.1,
                                       stroke_width=2.2)
            self.play(Create(box), run_time=0.5)
            self.wait(0.5)
            self.play(FadeOut(box), run_time=0.3)

        self.clear_stage(self.strip)

    def scene_compile(self):
        head, foot = self.begin(
            6, "Then ordinary code takes over",
            "Real predictions. The compiler resolves aliases; the model never saw them.",
        )
        self.play(self.strip.animate.move_to(UP * 1.6), run_time=0.5)

        tags = VGroup()
        for i, role in enumerate(PRED):
            if role == "O":
                continue
            shown = role[: -len("_CONT")] if role.endswith("_CONT") else role
            t = mono(shown, size=14, color=ROLE_COLOR.get(role, MUTED))
            t.next_to(self.strip[i], DOWN, buff=0.16)
            tags.add(t)

        alias = VGroup(
            mono("record", size=20, color=BLUE),
            mono("→", size=20, color=MUTED),
            mono("album", size=20, color=INK),
        ).arrange(RIGHT, buff=0.25)
        alias2 = VGroup(
            mono("performer", size=20, color=BLUE),
            mono("→", size=20, color=MUTED),
            mono("artist", size=20, color=INK),
        ).arrange(RIGHT, buff=0.25)
        aliases = VGroup(alias, alias2).arrange(DOWN, buff=0.22).shift(DOWN * 0.35)
        alias_label = mono("resolved on the CPU, from your schema", size=16,
                           color=MUTED).next_to(aliases, DOWN, buff=0.3)

        ast_text = json.dumps(DEMO["compiledAst"][0], separators=(", ", ": "))
        ast = mono(ast_text, size=17, color=INK)
        if ast.width > 12.0:
            ast.scale_to_fit_width(12.0)
        ast.next_to(alias_label, DOWN, buff=0.5)

        with self.voiceover(
            "The roles come out of the model. Then ordinary TypeScript takes over "
            "and resolves record to album, an alias the weights never saw, and "
            "builds the filter."
        ):
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tags],
                                  lag_ratio=0.12), run_time=1.3)
            self.play(LaggedStart(FadeIn(alias), FadeIn(alias2), lag_ratio=0.3),
                      run_time=1.1)
            self.play(FadeIn(alias_label), run_time=0.4)
            self.play(FadeIn(ast, shift=UP * 0.1), run_time=0.9)
            self.wait(0.6)

        self.clear_stage()

    def scene_result(self):
        head, foot = self.begin(
            7, "A schema it has never seen",
            "Generated evaluation corpus, four thousand queries. Not a claim about real user phrasing.",
        )

        trained = VGroup(
            mono("trained on", size=17, color=MUTED),
            mono("issues · mail · files · commits", size=22, color=INK),
        ).arrange(DOWN, buff=0.22)
        evaluated = VGroup(
            mono("evaluated on", size=17, color=MUTED),
            mono("contacts · music · recipes · shipments", size=22, color=BLUE),
        ).arrange(DOWN, buff=0.22)
        pair = VGroup(trained, evaluated).arrange(DOWN, buff=0.7).shift(UP * 1.35)

        shared = mono("shared vocabulary: none", size=19, color=AMBER).next_to(
            pair, DOWN, buff=0.5
        )

        number = mono("0.9888", size=68, color=INK).next_to(shared, DOWN, buff=0.45)
        number_label = mono("exact filter match", size=19, color=MUTED).next_to(
            number, DOWN, buff=0.18
        )

        with self.voiceover(
            "It was trained on issue trackers, mail, files and commits. It was "
            "evaluated on contacts, music, recipes and shipments, which share not "
            "one field word. Ninety eight point nine percent of queries compiled "
            "to exactly the right filter."
        ):
            self.play(FadeIn(trained, shift=UP * 0.15), run_time=0.8)
            self.play(FadeIn(evaluated, shift=UP * 0.15), run_time=0.8)
            self.play(FadeIn(shared), run_time=0.6)
            self.play(FadeIn(number, scale=0.85), FadeIn(number_label), run_time=1.1)
            self.wait(1.0)

        self.clear_stage()

    def scene_lockup(self):
        self.play(FadeOut(self.counter), run_time=0.3)
        mark = mono("gpu-query", size=52, color=INK).shift(UP * 0.5)
        facts = VGroup(
            mono(f"{CKPT['parameters']:,} parameters", size=22, color=MUTED),
            mono("runs on your machine  ·  no request leaves the page", size=22,
                 color=MUTED),
            mono("experimental", size=22, color=AMBER),
        ).arrange(DOWN, buff=0.24).next_to(mark, DOWN, buff=0.6)

        with self.voiceover(
            "G P U query. Twenty nine thousand parameters, running on your own "
            "machine. It is experimental, and built for learning."
        ):
            self.play(FadeIn(mark, scale=0.92), run_time=0.9)
            self.play(LaggedStart(*[FadeIn(f) for f in facts], lag_ratio=0.25),
                      run_time=1.4)
            self.wait(1.2)
        self.play(FadeOut(mark), FadeOut(facts), FadeOut(self.chrome), run_time=0.8)
