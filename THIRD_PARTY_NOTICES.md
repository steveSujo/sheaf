# Third-Party Notices

This file lists the third-party Rust crates compiled into `sheaf`, extracted from `Cargo.lock` via `cargo metadata --format-version 1`. Local workspace crates (`lean-*`) are excluded. Author-owned `orion` (`v0.1.0`, Apache-2.0 / MIT) is listed separately below.

## Author-Owned External Dependency

- **orion** $(@{name=orion; version=0.1.0; id=git+https://github.com/steveSujo/orion.git?tag=v0.1.0#orion@0.1.0; license=; license_file=; description=; source=git+https://github.com/steveSujo/orion.git?tag=v0.1.0#9f3702d5a6bde106b846ad429dcb949dad46fd97; dependencies=System.Object[]; targets=System.Object[]; features=; manifest_path=C:\Users\SteveSP\.cargo\git\checkouts\orion-b162c4bb54e39a9c\9f3702d\Cargo.toml; metadata=; publish=; authors=System.Object[]; categories=System.Object[]; keywords=System.Object[]; readme=README.md; repository=; homepage=; documentation=; edition=2021; links=; default_run=; rust_version=}.version) (Apache-2.0 OR MIT) - https://github.com/steveSujo/orion

## Third-Party Rust Crates (131 packages)

| Crate | Version | License (SPDX) | Authors / Maintainers | Repository |
| :--- | :--- | :--- | :--- | :--- |
| aead | 0.5.2 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| aes | 0.8.4 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/block-ciphers |
| aes-gcm | 0.10.3 | Apache-2.0 OR MIT | RustCrypto Developers | https://github.com/RustCrypto/AEADs |
| aho-corasick | 1.1.5 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/aho-corasick |
| argon2 | 0.5.3 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/password-hashes/tree/master/argon2 |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | Josh Stone <cuviper@gmail.com> | https://github.com/cuviper/autocfg |
| base64ct | 1.8.3 | Apache-2.0 OR MIT | RustCrypto Developers | https://github.com/RustCrypto/formats |
| bitflags | 2.13.2 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/bitflags/bitflags |
| bit-set | 0.11.1 | Apache-2.0 OR MIT | Alexis Beingessner <a.beingessner@gmail.com> | https://github.com/contain-rs/bit-vec |
| bit-vec | 0.10.2 | Apache-2.0 OR MIT | Alexis Beingessner <a.beingessner@gmail.com> | https://github.com/contain-rs/bit-vec |
| blake2 | 0.10.6 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/hashes |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| borsh | 1.8.1 | MIT OR Apache-2.0 | Near Inc <hello@near.org> | https://github.com/near/borsh-rs |
| borsh-derive | 1.8.1 | Apache-2.0 | Near Inc <hello@nearprotocol.com> | https://github.com/near/borsh-rs |
| bumpalo | 3.20.4 | MIT OR Apache-2.0 | Nick Fitzgerald <fitzgen@gmail.com> | https://github.com/fitzgen/bumpalo |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | Lokathor <zefria@gmail.com> | https://github.com/Lokathor/bytemuck |
| bytemuck_derive | 1.12.1 | Zlib OR Apache-2.0 OR MIT | Lokathor <zefria@gmail.com> | https://github.com/Lokathor/bytemuck |
| bytes | 1.12.2 | MIT | Carl Lerche <me@carllerche.com>, Sean McArthur <sean@seanmonstar.com> | https://github.com/tokio-rs/bytes |
| cfg_aliases | 0.2.2 | MIT | Zicklag <zicklag@katharostech.com> | https://github.com/katharostech/cfg_aliases |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | Alex Crichton <alex@alexcrichton.com> | https://github.com/rust-lang/cfg-if |
| chacha20 | 0.10.2 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/stream-ciphers |
| cipher | 0.4.4 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| console | 0.16.6 | MIT | - | https://github.com/console-rs/console |
| core_detect | 1.0.0 | MIT/Apache-2.0 | Thom Chiovoloni <chiovolonit@gmail.com> | https://github.com/thomcc/core_detect |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| cpufeatures | 0.3.1 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 | - | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 | - | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 | - | https://github.com/crossbeam-rs/crossbeam |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| ctr | 0.9.2 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/block-modes |
| digest | 0.10.7 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| either | 1.19.0 | MIT OR Apache-2.0 | - | https://github.com/rayon-rs/either |
| encode_unicode | 1.0.0 | Apache-2.0 OR MIT | Torbjørn Birch Moltu <t.b.moltu@lyse.net> | https://github.com/tormol/encode_unicode |
| env_filter | 2.1.0 | MIT OR Apache-2.0 | - | https://github.com/rust-cli/env_logger |
| env_logger | 0.11.12 | MIT OR Apache-2.0 | - | https://github.com/rust-cli/env_logger |
| equivalent | 1.0.3 | Apache-2.0 OR MIT | - | https://github.com/indexmap-rs/equivalent |
| errno | 0.3.15 | MIT OR Apache-2.0 | Chris Wong <lambda.fairy@gmail.com>, Dan Gohman <dev@sunfishcode.online> | https://github.com/lambda-fairy/rust-errno |
| fastrand | 2.5.0 | Apache-2.0 OR MIT | Stjepan Glavina <stjepang@gmail.com> | https://github.com/smol-rs/fastrand |
| fnv | 1.0.7 | Apache-2.0 / MIT | Alex Crichton <alex@alexcrichton.com> | https://github.com/servo/rust-fnv |
| futures-core | 0.3.34 | MIT OR Apache-2.0 | - | https://github.com/rust-lang/futures-rs |
| futures-task | 0.3.34 | MIT OR Apache-2.0 | - | https://github.com/rust-lang/futures-rs |
| futures-util | 0.3.34 | MIT OR Apache-2.0 | - | https://github.com/rust-lang/futures-rs |
| generic-array | 0.14.7 | MIT | Bartłomiej Kamiński <fizyk20@gmail.com>, Aaron Trent <novacrazy@gmail.com> | https://github.com/fizyk20/generic-array.git |
| getrandom | 0.2.17 | MIT OR Apache-2.0 | The Rand Project Developers | https://github.com/rust-random/getrandom |
| getrandom | 0.4.3 | MIT OR Apache-2.0 | The Rand Project Developers | https://github.com/rust-random/getrandom |
| ghash | 0.5.1 | Apache-2.0 OR MIT | RustCrypto Developers | https://github.com/RustCrypto/universal-hashes |
| half | 1.8.3 | MIT OR Apache-2.0 | Kathryn Long <squeeself@gmail.com> | https://github.com/starkat99/half-rs |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | - | https://github.com/rust-lang/hashbrown |
| indexmap | 2.14.2 | Apache-2.0 OR MIT | - | https://github.com/indexmap-rs/indexmap |
| indicatif | 0.18.6 | MIT | - | https://github.com/console-rs/indicatif |
| inout | 0.1.4 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| itoa | 1.0.19 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/itoa |
| js-sys | 0.3.107 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/js-sys |
| libc | 0.2.191 | MIT OR Apache-2.0 | - | https://github.com/rust-lang/libc |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | Dan Gohman <dev@sunfishcode.online> | https://github.com/sunfishcode/linux-raw-sys |
| log | 0.4.35 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-lang/log |
| matrixmultiply | 0.3.11 | MIT/Apache-2.0 | bluss, R. Janis Goldschmidt | https://github.com/bluss/matrixmultiply/ |
| memchr | 2.8.3 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com>, bluss | https://github.com/BurntSushi/memchr |
| memmap2 | 0.5.10 | MIT OR Apache-2.0 | Dan Burkert <dan@danburkert.com>, Yevhenii Reizner <razrfalcon@gmail.com> | https://github.com/RazrFalcon/memmap2-rs |
| ndarray | 0.15.6 | MIT OR Apache-2.0 | Ulrik Sverdrup "bluss", Jim Turner | https://github.com/rust-ndarray/ndarray |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-num/num-complex |
| num-integer | 0.1.47 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-num/num-integer |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | The Rust Project Developers | https://github.com/rust-num/num-traits |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | Aleksey Kladov <aleksey.kladov@gmail.com> | https://github.com/matklad/once_cell |
| opaque-debug | 0.3.1 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/utils |
| password-hash | 0.5.0 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits/tree/master/password-hash |
| pin-project-lite | 0.2.18 | Apache-2.0 OR MIT | - | https://github.com/taiki-e/pin-project-lite |
| polyval | 0.6.2 | Apache-2.0 OR MIT | RustCrypto Developers | https://github.com/RustCrypto/universal-hashes |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT | - | https://github.com/taiki-e/portable-atomic |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | The CryptoCorrosion Contributors | https://github.com/cryptocorrosion/cryptocorrosion |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com>, Alex Crichton <alex@alexcrichton.com> | https://github.com/dtolnay/proc-macro2 |
| proc-macro-crate | 3.5.0 | MIT OR Apache-2.0 | Bastian Köcher <git@kchr.de> | https://github.com/bkchr/proc-macro-crate |
| proptest | 1.12.0 | MIT OR Apache-2.0 | Jason Lingle | https://github.com/proptest-rs/proptest |
| quickcheck | 1.1.0 | Unlicense OR MIT | Andrew Gallant <jamslam@gmail.com> | https://github.com/BurntSushi/quickcheck |
| quick-error | 1.2.3 | MIT/Apache-2.0 | Paul Colomiets <paul@colomiets.name>, Colin Kiegel <kiegel@gmx.de> | http://github.com/tailhook/quick-error |
| quote | 1.0.47 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/quote |
| rand | 0.10.4 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rand |
| rand | 0.8.8 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rand |
| rand_chacha | 0.3.1 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers, The CryptoCorrosion Contributors | https://github.com/rust-random/rand |
| rand_core | 0.10.1 | MIT OR Apache-2.0 | The Rand Project Developers | https://github.com/rust-random/rand_core |
| rand_core | 0.6.4 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rand |
| rand_xorshift | 0.5.0 | MIT OR Apache-2.0 | The Rand Project Developers, The Rust Project Developers | https://github.com/rust-random/rngs |
| rawpointer | 0.2.1 | MIT/Apache-2.0 | bluss | https://github.com/bluss/rawpointer/ |
| rayon | 1.12.0 | MIT OR Apache-2.0 | - | https://github.com/rayon-rs/rayon |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | - | https://github.com/rayon-rs/rayon |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | - | https://github.com/r-efi/r-efi |
| regex | 1.13.2 | MIT OR Apache-2.0 | The Rust Project Developers, Andrew Gallant <jamslam@gmail.com> | https://github.com/rust-lang/regex |
| regex-automata | 0.4.19 | MIT OR Apache-2.0 | The Rust Project Developers, Andrew Gallant <jamslam@gmail.com> | https://github.com/rust-lang/regex |
| regex-syntax | 0.8.12 | MIT OR Apache-2.0 | The Rust Project Developers, Andrew Gallant <jamslam@gmail.com> | https://github.com/rust-lang/regex |
| rustix | 1.1.6 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | Dan Gohman <dev@sunfishcode.online>, Jakub Konka <kubkon@jakubkonka.com> | https://github.com/bytecodealliance/rustix |
| rustversion | 1.0.23 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/rustversion |
| rusty-fork | 0.3.1 | MIT/Apache-2.0 | Jason Lingle | https://github.com/altsysrq/rusty-fork |
| serde | 1.0.229 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_cbor | 0.11.2 | MIT/Apache-2.0 | Pyfisch <pyfisch@posteo.org>, Steven Fackler <sfackler@gmail.com> | https://github.com/pyfisch/cbor |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | Erick Tryzelaar <erick.tryzelaar@gmail.com>, David Tolnay <dtolnay@gmail.com> | https://github.com/serde-rs/serde |
| serde_json | 1.0.154 | MIT OR Apache-2.0 | - | https://github.com/serde-rs/json |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/hashes |
| slab | 0.4.12 | MIT | Carl Lerche <me@carllerche.com> | https://github.com/tokio-rs/slab |
| subtle | 2.6.1 | BSD-3-Clause | Isis Lovecruft <isis@patternsinthevoid.net>, Henry de Valence <hdevalence@hdevalence.ca> | https://github.com/dalek-cryptography/subtle |
| syn | 2.0.119 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/syn |
| syn | 3.0.10 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/syn |
| tempfile | 3.27.0 | MIT OR Apache-2.0 | Steven Allen <steven@stebalien.com>, The Rust Project Developers, Ashley Mannix <ashleymannix@live.com.au>, Jason White <me@jasonwhite.io> | https://github.com/Stebalien/tempfile |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror | 2.0.21 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/thiserror |
| toml_datetime | 1.1.3+spec-1.1.0 | MIT OR Apache-2.0 | - | https://github.com/toml-rs/toml |
| toml_edit | 0.25.20+spec-1.1.0 | MIT OR Apache-2.0 | - | https://github.com/toml-rs/toml |
| toml_parser | 1.1.6+spec-1.1.0 | MIT OR Apache-2.0 | - | https://github.com/toml-rs/toml |
| typenum | 1.20.1 | MIT OR Apache-2.0 | - | https://github.com/paholg/typenum |
| unarray | 0.1.4 | MIT OR Apache-2.0 | - | https://github.com/cameron1024/unarray |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/unicode-ident |
| unicode-width | 0.2.3 | MIT OR Apache-2.0 | kwantam <kwantam@gmail.com>, Manish Goregaokar <manishsmail@gmail.com> | https://github.com/unicode-rs/unicode-width |
| unit-prefix | 0.5.2 | MIT | Fabio Valentini <decathorpe@gmail.com>, Benjamin Sago <ogham@bsago.me> | https://codeberg.org/commons-rs/unit-prefix |
| universal-hash | 0.5.1 | MIT OR Apache-2.0 | RustCrypto Developers | https://github.com/RustCrypto/traits |
| version_check | 0.9.5 | MIT/Apache-2.0 | Sergio Benitez <sb@sergio.bz> | https://github.com/SergioBenitez/version_check |
| wait-timeout | 0.2.1 | MIT/Apache-2.0 | Alex Crichton <alex@alexcrichton.com> | https://github.com/alexcrichton/wait-timeout |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | The Cranelift Project Developers | https://github.com/bytecodealliance/wasi |
| wasm-bindgen | 0.2.130 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen |
| wasm-bindgen-macro | 0.2.130 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/macro |
| wasm-bindgen-macro-support | 0.2.130 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/main/crates/macro-support |
| wasm-bindgen-shared | 0.2.130 | MIT OR Apache-2.0 | The wasm-bindgen Developers | https://github.com/wasm-bindgen/wasm-bindgen/tree/master/crates/shared |
| web-time | 1.1.0 | MIT OR Apache-2.0 | - | https://github.com/daxpedda/web-time |
| windows-link | 0.2.1 | MIT OR Apache-2.0 | - | https://github.com/microsoft/windows-rs |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 | - | https://github.com/microsoft/windows-rs |
| winnow | 1.0.5 | MIT | - | https://github.com/winnow-rs/winnow |
| zerocopy | 0.8.64 | BSD-2-Clause OR Apache-2.0 OR MIT | - | https://github.com/google/zerocopy |
| zerocopy-derive | 0.8.64 | BSD-2-Clause OR Apache-2.0 OR MIT | - | https://github.com/google/zerocopy |
| zmij | 1.0.23 | MIT | David Tolnay <dtolnay@gmail.com> | https://github.com/dtolnay/zmij |
