const SMA = require("./tools/SMA");
const EMA = require("./tools/EMA");
const WMA = require("./tools/WMA");
const MMA = require("./tools/MMA");

const predef = require("./tools/predef");
const meta = require("./tools/meta");
const medianPrice = require("./tools/medianPrice");
const typicalPrice = require("./tools/typicalPrice");

const NAMED = {
    red:[255,0,0], blue:[0,0,255], green:[0,128,0], orange:[255,165,0],
    yellow:[255,255,0], purple:[128,0,128], cyan:[0,255,255],
    magenta:[255,0,255], black:[0,0,0], white:[255,255,255]
};

function parseRGB(c) {
    if (!c) return [255,255,255];
    if (NAMED[c]) return NAMED[c];
    if (c.startsWith("#")) {
        const hex = c.slice(1);
        if (hex.length === 3) {
            const r = parseInt(hex[0]+hex[0],16);
            const g = parseInt(hex[1]+hex[1],16);
            const b = parseInt(hex[2]+hex[2],16);
            return [r,g,b];
        } else if (hex.length === 6) {
            return [
                parseInt(hex.slice(0,2),16),
                parseInt(hex.slice(2,4),16),
                parseInt(hex.slice(4,6),16)
            ];
        }
    }
    // rudimentary rgb(...) fallback
    const m = c.match(/rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
    return [255,255,255];
}

function mixWithWhite(rgb, t) {
    // t in [0,1]: 0 => original, 1 => white
    const [r,g,b] = rgb;
    const R = Math.round(r*(1-t) + 255*t);
    const G = Math.round(g*(1-t) + 255*t);
    const B = Math.round(b*(1-t) + 255*t);
    return `rgb(${R}, ${G}, ${B})`;
}

function rgba(rgb, a) {
    const [r,g,b] = rgb;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

class TrendHeat {
    init() {
        this.sma = SMA(this.props.period);
        this.ema = EMA(this.props.period);
        this.wma = WMA(this.props.period);
        this.mma = MMA(this.props.period);

        // trend state
        this._streak = 0;     // bars since last cross
        this._lastSide = 0;   // -1 below MA, +1 above MA, 0 unknown
    }

    _shade(baseColor, streak) {
        // Normalize streak -> [0,1] fade progress
        const L = Math.max(1, this.props.fadeLength || 10);
        const progress = L === 1 ? 1 : Math.min((streak - 1) / (L - 1), 1);

        const baseRGB = parseRGB(baseColor);

        if (this.props.fadeMode === "alpha") {
            // NEW: newer bars more transparent; older bars more opaque
            const alphaMin = 0.25;  // transparency at the cross (more see-through)
            const alphaMax = 1.00;  // opacity after a long streak
            const alpha = alphaMin + (alphaMax - alphaMin) * progress; // 0.25 -> 1.0
            return rgba(baseRGB, Math.max(0.05, Math.min(1, alpha)));
        } else {
            // LIGHTEN mode unchanged (blend toward white as streak grows)
            const tMax = 0.85;
            const t = progress * tMax;
            return mixWithWhite(baseRGB, t);
        }
    }

    map(d) {
        // pick price source
        const close = d.close();
        const price = this.props.price === "hl2"
            ? medianPrice(d)
            : this.props.price === "hlc3"
            ? typicalPrice(d)
            : close;

        // MA value BEFORE feeding current price (classic)
        const average = this[this.props.average].avg() || price;

        // update MA with current price
        this[this.props.average](price);

        // Determine side vs MA
        const side = close > average ? 1 : (close < average ? -1 : 0);

        if (side === 0) {
            // do not reset on exact tie; keep streak/side
        } else if (side === this._lastSide) {
            // continuing trend
            this._streak += 1;
        } else {
            // new cross -> reset
            this._streak = 1;
            this._lastSide = side;
        }

        // choose base color & shade it by streak
        let color = "white";
        if (this._lastSide > 0) {
            color = this._shade(this.props.trendUp, Math.max(1, this._streak));
        } else if (this._lastSide < 0) {
            color = this._shade(this.props.trendDown, Math.max(1, this._streak));
        }

        return {
            candlestick: { color },
            value: average
        };
    }
}

module.exports = {
    name: "TrendHeat",
    description: "TrendHeat (trend-length fade on candles)",
    calculator: TrendHeat,
    params: {
        period: predef.paramSpecs.period(6),
        trendUp: predef.paramSpecs.color("blue"),
        trendDown: predef.paramSpecs.color("red"),
        price: predef.paramSpecs.enum({
            hl2: "High + Low / 2",
            hlc3: "High + Low + Close / 3",
            close: "Close"
        }, "hl2"),
        average: predef.paramSpecs.enum({
            sma: "Simple",
            ema: "Exponential",
            wma: "Weighted",
            mma: "Modified"
        }, "sma"),
        // NEW
        fadeLength: predef.paramSpecs.period(10), // bars to reach lightest shade
        fadeMode: predef.paramSpecs.enum({ lighten: "Lighten", alpha: "Opacity" }, "lighten")
    },
    tags: ["tystr"],
    inputType: meta.InputType.BARS,
    schemeStyles: predef.styles.solidLine("white")
};
