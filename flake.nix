{
  description = "centroid-gap: Lean 4 proofs about IVF centroid routing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          py = pkgs.python3.withPackages (ps: [ ps.numpy ps.faiss ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.elan pkgs.uv pkgs.git py ];
          };
        });
    };
}
